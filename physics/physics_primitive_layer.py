"""
Physics Primitive Layer — LTE-TrackGuard
=========================================

Tiga primitif collision yang dimensionless, scale-free, dan tracker-agnostic.
Diturunkan dari first principles — bukan threshold engineering.

PRIMITIVES:
  CE              — Contact Event (kinematic contact definition)
  KELR_personal   — Kinetic Energy Loss Ratio, per-track scalar (crash physics)
  ARS             — Aspect Ratio Shock, per-track statistical (hypothesis testing)

TRIGGER LOGIC (Blueprint A8):
  COLLISION = (CE_confirmed AND (KELR_personal OR ARS) AND partner_mobile)
           OR (KELR_personal AND IoU > 0)

  CE_confirmed (REDESAIN: satu kenaikan IoU, bukan dua):
    Frame N  : CE_raw fires (IoU 0→positif + approach_rate + rel_speed) → PENDING stage=0
    Frame N+1: iou_now > pending_iou  → CE CONFIRMED langsung (1 rise, bukan 2)
               iou_now ≤ pending_iou  → DISCARDED (papasan / belok sekilas)
    TTL expired                       → DISCARDED
    (Alasan redesain: data empiris menunjukkan 2-stage membunuh 88% sinyal valid
     karena post-collision IoU plateau/turun — bukan naik 2x berturut-turut)

  CE+ARS guards (hanya berlaku jika KELR tidak fire):
    A8.1 — Partner mobility: kedua track harus net_disp > bbox_diag/4 dalam W.
            Mencegah FP "bus lewat dekat post-collision wreck" (wreck net_disp ≈ 0).
    A8.2 — Papasan filter: cos(θ) < -0.5 (angle > 120°, berlawanan arah) → SUPPRESS.
            Physics: tabrakan head-on nyata → deceleration masif → KELR fires → path lain.
            CE+ARS tanpa KELR + arah berlawanan = papasan saja.
            Angle dihitung dari net displacement vector (W-frame), bukan instantaneous
            velocity — robust terhadap jitter deteksi single-frame.

  ARS was_moving guard (internal, per-track):
    Menggunakan net_displacement (bukan total_path_length) agar jitter deteksi
    tidak membuat kendaraan diam dianggap "moving".

  Tabrakan nyata  : IoU naik ≥ 2x berturut-turut   → CE confirmed ✓
  Crossing sekilas: IoU naik 1x lalu drop ke 0     → CE discarded ✓
  Papasan/belok   : IoU kembali ke 0 tanpa rise 2x  → CE discarded ✓
  Proximity sekilas: IoU turun atau stabil di nol   → CE discarded ✓
  Bus + wreck diam: partner net_disp ≈ 0             → A8.1 suppress ✓
  Papasan berlawanan: cos(θ) < -0.5, no KELR        → A8.2 suppress ✓

  KELR_personal: per-track scalar speed drop, butuh proximity guard (IoU > 0).
  ARS: deformation/roll shock, has internal was_moving guard (net_disp-based).

TRACKER CONTRACT:
  Layer ini hanya membutuhkan dari tracker:
    - track.track_id (int)
    - track.history  (list of dict, masing-masing punya 'center', 'bbox')
    - track.hits     (int)
    - track.current_detection (dict dengan 'center', 'bbox')

  Tidak bergantung pada velocity smoothed tracker, Kalman prediction,
  atau threshold apapun dari tracker.

REFERENSI:
  - Brach, R.M. (1991). Mechanical Impact Dynamics. Wiley.
    → KELR threshold = 1 - e² = 0.91, e = 0.3 untuk moderate crash
  - Statistical hypothesis testing standard (2σ = 95% confidence)
    → ARS threshold = 2.0σ
  - Perception-reaction time dari accident reconstruction literature
    → W = fps × 1.0 detik
"""

import numpy as np
import math
from typing import List, Dict, Optional, Tuple


class PhysicsPrimitiveLayer:
    """
    Tracker-agnostic collision detector berbasis tiga physics primitives.

    Gunakan satu instance per video session. Reset antar video dengan reset().
    FPS bisa di-update mid-session via update_fps().
    """

    # ── Physics constants (derived from literature, NOT tuned) ──────────────
    COEFF_RESTITUTION  = 0.346                      # e, dari Brach (1991); lowered from 0.3
    KELR_THRESHOLD     = 1.0 - (0.346 ** 2)        # = 0.880, dari 1 - e²
    CE_MIN_SPEED_RATIO = 0.15                       # dimensionless relative speed guard
    KAPPA_PROX = 2.0                                # KELR proximity multiplier
    KAPPA_PROX_STRICT = 1.2                         # strict-scene proximity multiplier
    KELR_IOU_DELTA_MIN = 0.02                       # Path C IoU-delta guard
    STRICT_SINGLE_KELR_IOU_DELTA_MIN = 0.03         # strict-scene single-KELR IoU-delta guard
    ARS_SIGMA          = 2.0                        # 95% confidence (2σ)
    AR_BASELINE_N      = 20                         # frames untuk estimasi AR mean/std
    # ────────────────────────────────────────────────────────────────────────

    # ── KELR guard: EAGER-inspired statistical anomaly test ──────────────────
    # Menggantikan noise_floor pixel-based dengan Z-score self-normalizing.
    #
    # Konsep dari EAGER (Andriyadi et al.): temporal anomaly diukur bukan
    # dengan threshold absolut, tapi dengan seberapa jauh nilai saat ini
    # menyimpang dari distribusi historisnya sendiri.
    #
    # Implementasi:
    #   speed_after = mean(||v_rel||, last KELR_N_AFTER frames)   ← EAGER multi-frame avg
    #   z_drop = (mean_speed_before - speed_after) / std_speed_before
    #   guard  = z_drop > ARS_SIGMA (= 2σ)                        ← 95% confidence
    #
    # std_speed_before mengabsorb level jitter aktual video tsb —
    # tidak ada magic number, tidak ada asumsi resolusi kamera.
    KELR_N_AFTER = 3   # frames untuk rata-rata v_rel_after (multi-frame, bukan single)

    # ── Path D: Persistent IoU contact + kinematic evidence (slow-speed collision) ─
    # Redesign: IoU persistence saja tidak cukup — harus ada bukti energi selama
    # streak berlangsung. Ini membedakan tabrakan fisik (ada deceleration/spin)
    # dari overlap perspektif atau berhenti di lampu merah (tidak ada energi event).
    #
    # Syarat baru: dalam window streak, setidaknya satu track harus pernah punya:
    #   KELR_val > PATH_D_KELR_MIN  (deceleration, threshold longgar)
    #   OR ARS_z  > PATH_D_ARS_MIN  (aspect ratio shock, threshold longgar)
    #
    # Threshold sengaja di BAWAH threshold utama (0.91 / 2.0) agar Path-D bisa
    # tangkap collision energi lemah yang tidak lolos Path A/B/C.
    IOU_PERSIST_MIN      = 0.10  # minimum IoU untuk dihitung sebagai kontak persisten
    IOU_PERSIST_N        = 5     # jumlah frame berturut-turut yang diperlukan
    PATH_D_KELR_MIN      = 0.5   # KELR minimum selama streak (lebih longgar dari 0.91)
    PATH_D_ARS_MIN       = 1.5   # ARS z-score minimum selama streak (lebih longgar dari 2.0)
    PATH_D_RECENT_DECAY  = 0.88  # decay rolling KELR/ARS per frame (0.91 → <0.5 dalam ~5 frame)
    POST_CE_WINDOW       = 5     # frame tambahan untuk KELR/ARS fire setelah CE confirmed

    # ── Path E: Dual Kinematic Burst (safety net tanpa IoU persistence) ──────
    # Menangkap tabrakan di mana KELR dan ARS keduanya tinggi secara bersamaan
    # meski IoU terlalu singkat untuk Path D atau CE terlalu sebentar untuk Path A.
    # Khas: sideswipe oblique, tabrakan singkat, bounding box barely overlap.
    # Guard: direction (bukan antiparallel) + both_moving + CE_raw proximity.
    PATH_E_KELR_MIN  = 0.82   # min KELR_val — dekat threshold utama (0.91), hindari deselerasi normal
    PATH_E_ARS_MIN   = 1.8    # min ARS_z — dekat threshold utama (2.0σ), hindari cornering biasa
    PATH_E_BURST_N   = 5      # frame berturutan — cukup lama untuk bedakan impact vs noise
    PATH_E_CE_WINDOW = 5      # TTL frame CE_raw proximity — ketat, hanya capture kontak nyata

    # ── Path PMV: Pairwise Momentum Violation ────────────────────────────────
    # Prinsip: tabrakan = transfer momentum impulsif antara dua track.
    # p_sys(t) = area_i × speed_i(t) + area_j × speed_j(t)
    # Jika p_sys runtuh mendadak (z_drop > 2σ) saat kedua track berdekatan → collision.
    # Normal braking: p_sys turun bertahap → z rendah. Tabrakan: p_sys runtuh tiba-tiba.
    # Proxy massa = bbox_area (scale-free, tidak butuh kalibrasi kamera).
    PMV_MIN_REL_CHANGE = 0.15  # minimal 15% perubahan momentum relatif pada KEDUA track

    # CE confirmation window (REDESAIN: 1-stage, bukan 2-stage)
    # Max frames untuk menunggu IoU deepening setelah CE_raw fires.
    # Physics: brief grazing = IoU kembali ke 0 dalam 1-2 frame.
    #          Collision nyata = IoU naik minimal 1x dalam window ini.
    # Data empiris: 2-stage confirmation membunuh 88% sinyal valid (CE_raw fire 98.3%
    # tapi CE_confirmed hanya 11.9% karena post-impact IoU plateau, bukan naik 2x).
    # Solusi: 1 rise saja sudah cukup untuk confirming deepening contact.
    CE_CONFIRM_WINDOW = 4     # sedikit diperbesar karena sekarang 1-stage (lebih toleran)

    # ── Path A Sliding Window (REDESAIN) ──────────────────────────────────────
    # CE_raw dan KELR tidak harus terjadi dalam 1 frame yang sama (co-occurrence ketat).
    # Data empiris: gap KELR-CE_raw tersebar dari -62 s/d +183 frame (mean 9.5 frame).
    # Solusi: jika CE_raw + KELR terjadi dalam PATH_A_WINDOW frame apapun → Path A Sliding.
    # Guard: IoU minimum 0.05 (overlap bbox nyata), bukan sekadar center proximity.
    # Alasan: proximity berbasis center_dist terlalu longgar di highway oblique —
    # kendaraan di lane sebelah qualify sebagai "proximity" tanpa overlap fisik.
    PATH_A_WINDOW = 10        # 10 frame ≈ 333ms @ 30fps — cukup menangkap gap timing
    PATH_A_SW_MIN_IOU = 0.05  # minimum IoU untuk A_sw — harus ada overlap bbox nyata
    PATH_A_KELR_MIN = 0.91    # KELR threshold penuh (tidak dilonggarkan — presisi tetap terjaga)
    # ARS-only guard untuk CE path (Fix 2a/2b):
    # Di kamera oblique highway, ARS 2.0-3.0σ sering dari perubahan perspektif, bukan deformasi.
    # Jika KELR tidak fire, ARS hanya valid sebagai bukti CE jika:
    #   (a) IoU >= 0.05 (ada overlap bbox nyata), DAN
    #   (b) ARS >= 3.0σ (cukup ekstrem untuk bukan sekadar perspektif)
    ARS_CE_SIGMA_HIGH = 3.0   # threshold ARS untuk CE+ARS-only path (tanpa KELR)
    # ────────────────────────────────────────────────────────────────────────

    def __init__(self, fps: float = 30.0, config: Optional[Dict] = None):
        """
        Args:
            fps: Frame rate video. Dipakai untuk menentukan window W = fps × 1s.
        """
        config = config or {}
        self.CE_MIN_SPEED_RATIO = float(config.get('rho_min', self.CE_MIN_SPEED_RATIO))
        self.KAPPA_PROX = float(config.get('kappa_prox', self.KAPPA_PROX))
        self.KAPPA_PROX_STRICT = float(config.get('kappa_prox_strict', self.KAPPA_PROX_STRICT))
        self.KELR_IOU_DELTA_MIN = float(config.get('delta_iou', self.KELR_IOU_DELTA_MIN))
        self.KELR_THRESHOLD = float(config.get('tau_kelr', self.KELR_THRESHOLD))
        self.PATH_A_KELR_MIN = float(config.get('tau_kelr', self.PATH_A_KELR_MIN))
        self.ARS_SIGMA = float(config.get('tau_ars', self.ARS_SIGMA))
        self.KELR_Z_THRESHOLD = float(config.get('tau_z', self.ARS_SIGMA))
        self.STRICT_SINGLE_KELR_IOU_DELTA_MIN = float(
            config.get('strict_single_kelr_delta_iou', self.STRICT_SINGLE_KELR_IOU_DELTA_MIN)
        )
        primitive_mask_raw = config.get('primitive_mask', 'ce,kelr,ars,pmv')
        if isinstance(primitive_mask_raw, str):
            primitive_mask = {p.strip().lower() for p in primitive_mask_raw.split(',') if p.strip()}
        else:
            primitive_mask = {str(p).strip().lower() for p in primitive_mask_raw}
        if 'all' in primitive_mask:
            primitive_mask = {'ce', 'kelr', 'ars', 'pmv'}
        self.enable_primitive_ce = 'ce' in primitive_mask
        self.enable_primitive_kelr = 'kelr' in primitive_mask
        self.enable_primitive_ars = 'ars' in primitive_mask
        self.enable_primitive_pmv = 'pmv' in primitive_mask
        self.primitive_mask = ','.join(sorted(primitive_mask)) if primitive_mask else 'none'
        self.fps = fps
        self._update_window()

        # IoU history per pair untuk deteksi transisi CE (0 → positif)
        # key: (min_id, max_id) tuple agar simetris
        self._iou_history: Dict[Tuple[int, int], List[float]] = {}

        # CE pending state untuk deepening contact confirmation (Blueprint A7, two-rise)
        # key: (min_id, max_id) → dict dengan:
        #   'iou'      : float — IoU saat CE_raw fire (stage=0 baseline)
        #   'stage'    : int   — 0 = menunggu rise pertama, 1 = menunggu rise kedua
        #   'rise_iou' : float — IoU saat rise pertama terjadi (stage=1 baseline)
        #   'kelr'     : bool  — apakah KELR fire selama pending window (accumulated)
        #   'ars'      : bool  — apakah ARS fire selama pending window (accumulated)
        #   'ttl'      : int   — sisa frame sebelum pending expired (CE_CONFIRM_WINDOW)
        # Accumulated evidence: kinematic signal dari frame CE_raw atau frame berikutnya
        # diwariskan ke trigger saat CE dikonfirmasi — mencegah "wasted evidence".
        self._pending_ce: Dict[Tuple[int, int], Dict] = {}

        # Path D: IoU contact streak counter + kinematic evidence tracker
        # key: (min_id, max_id) → int jumlah frame berturut-turut dengan IoU > IOU_PERSIST_MIN
        self._iou_contact_streak: Dict[Tuple[int, int], int] = {}
        # Max KELR dan ARS yang teramati selama streak IoU berlangsung (per pair).
        # Direset saat streak putus. Dipakai sebagai kinematic guard di Path-D.
        self._iou_streak_kelr_max: Dict[Tuple[int, int], float] = {}
        self._iou_streak_ars_max:  Dict[Tuple[int, int], float] = {}
        # Rolling max KELR/ARS per pair — diupdate SETIAP frame dengan decay eksponensial.
        # Menangkap kinematic evidence SEBELUM streak IoU dimulai (pre-streak lookback).
        # Decay: 0.88/frame → nilai 0.91 jadi < 0.5 dalam ~5 frame (window efektif ~5 frame).
        self._pair_kelr_recent: Dict[Tuple[int, int], float] = {}
        self._pair_ars_recent:  Dict[Tuple[int, int], float] = {}
        # Post-CE window: setelah CE confirmed, beri KELR/ARS POST_CE_WINDOW frame tambahan.
        # Menangkap kasus V7 di mana KELR fire 3 frame SETELAH CE confirmed.
        # key: pair_key → sisa frame dalam window
        self._post_ce_window: Dict[Tuple[int, int], int] = {}

        # ── Path A Sliding Window state (REDESAIN) ───────────────────────────
        # Menyimpan frame terakhir CE_raw fire dan KELR fire per pair.
        # Jika keduanya ada dalam PATH_A_WINDOW frame → trigger Path A Sliding.
        # key: pair_key → frame_id saat last CE_raw fire
        self._ce_raw_last_frame: Dict[Tuple[int, int], int] = {}
        # key: pair_key → frame_id saat last KELR fire (kelr_i or kelr_j = True)
        self._kelr_last_frame: Dict[Tuple[int, int], int] = {}

        # ── Path E state ─────────────────────────────────────────────────────
        # Burst counter: frame berturutan di mana KELR+ARS keduanya di atas threshold.
        self._path_e_burst: Dict[Tuple[int, int], int] = {}
        # CE_raw proximity: TTL counter — di-set ke PATH_E_CE_WINDOW saat ce_raw=True.
        # Memungkinkan CE_raw yang terjadi sebelum burst (pre-contact) tetap dipakai.
        self._path_e_ce_recent: Dict[Tuple[int, int], int] = {}

        # ── Scalar logging (discriminability evaluation, IEEE T-IM) ────────────
        # Disabled by default. Enable via enable_scalar_logging() dari main.py.
        # Mencatat nilai mentah semua proximate pair yang dievaluasi — baik yang
        # trigger maupun tidak — untuk analisis ROC/AUC per-primitive.
        self._scalar_log_enabled: bool = False
        self._scalar_log_video: str = ""
        self._scalar_log_label: int = 0        # 0=non-collision, 1=collision
        self._scalar_log_angle: str = "unknown" # 'oblique' | 'overhead' | 'unknown'
        self._scalar_log: list = []
        # Scene density — diupdate per-frame via update_scene_info()
        self._scene_density: float = 0.0
        self._scene_category: str = "unknown"
        # ────────────────────────────────────────────────────────────────────

        print(f"✓ PhysicsPrimitiveLayer initialized")
        print(f"  FPS={fps:.1f}  |  W={self.W} frames (1 detik)")
        print(f"  KELR threshold  : {self.KELR_THRESHOLD:.3f}  (e={self.COEFF_RESTITUTION})")
        print(f"  ARS  threshold  : {self.ARS_SIGMA}σ  (95% confidence)")
        print(f"  CE   min_ratio  : {self.CE_MIN_SPEED_RATIO}  (dimensionless)")
        print(f"  KELR proximity  : kappa={self.KAPPA_PROX} | strict={self.KAPPA_PROX_STRICT}")
        print(f"  Path C delta IoU: {self.KELR_IOU_DELTA_MIN} | strict single={self.STRICT_SINGLE_KELR_IOU_DELTA_MIN}")
        print(f"  KELR guard      : Z-score > {self.ARS_SIGMA}σ (EAGER-inspired, self-normalizing)")
        print(f"  Trigger logic   : PathA:(CE+KELR/ARS) OR PathB:(mutual_KELR+prox+dir) OR PathC:(KELR+IoU>0+dir)  [Blueprint A8]")
        print(f"  Primitive mask  : {self.primitive_mask}")
        print(f"  CE confirmation : two-rise deepening (IoU naik 2x berturut-turut dalam {self.CE_CONFIRM_WINDOW} frames, evidence accumulated)")
        print(f"  A8.1 partner mob: net_disp > bbox_diag/4 (both tracks, CE+ARS path only)")
        print(f"  A8.2 papasan flt: cos(θ) < -0.5 → angle > 120° → suppress (CE+ARS, no KELR) [net_disp_vec, W-frame]")

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════════════════
    # SCALAR LOGGING — discriminability evaluation (IEEE T-IM)
    # ══════════════════════════════════════════════════════════════════════════

    def enable_scalar_logging(self, video_name: str, label: int = 0,
                              camera_angle: str = 'unknown') -> None:
        """Aktifkan scalar logging untuk evaluasi discriminability.

        Panggil SEBELUM memulai pemrosesan video.

        Args:
            video_name   : Nama file video (dipakai sebagai key di CSV).
            label        : Ground truth label — 0=non-collision, 1=collision.
            camera_angle : Tipe sudut kamera — 'oblique', 'overhead', atau 'unknown'.
        """
        self._scalar_log_enabled = True
        self._scalar_log_video   = video_name
        self._scalar_log_label   = int(label)
        self._scalar_log_angle   = camera_angle
        self._scalar_log.clear()

    def disable_scalar_logging(self) -> None:
        """Nonaktifkan scalar logging tanpa menghapus log yang sudah terkumpul."""
        self._scalar_log_enabled = False

    def update_scene_info(self, density: float, category: str) -> None:
        """Update scene density per-frame. Dipanggil dari collision_detector sebelum check_pair."""
        self._scene_density = density
        self._scene_category = category

    def get_scalar_log(self) -> list:
        """Kembalikan salinan log yang terkumpul (list of dict)."""
        return list(self._scalar_log)

    def clear_scalar_log(self) -> None:
        """Hapus log yang terkumpul."""
        self._scalar_log.clear()

    # ══════════════════════════════════════════════════════════════════════════

    def check_pair(self, track_i, track_j, frame_id: int) -> Optional[Dict]:
        """
        Periksa sepasang track untuk collision menggunakan Physics Primitive Layer.

        Panggil sekali per pasangan track per frame dari collision_detector.

        Args:
            track_i, track_j : Track objects (SHA atau ByteTrack, tidak masalah)
            frame_id         : Frame saat ini

        Returns:
            Dict hasil collision jika triggered, None jika tidak.
        """
        # Minimal satu track harus punya history cukup.
        # Setiap primitive (_compute_KELR_personal, _compute_CE, _compute_ARS)
        # sudah handle insufficient history dengan graceful return False,
        # sehingga guard AND berlebihan dan memblokir deteksi saat partner
        # adalah track baru yang valid (e.g., korban tabrakan baru muncul di frame).
        if not (self._has_enough_history(track_i) or
                self._has_enough_history(track_j)):
            return None

        # ── Guard: exclude truck/bus from PPL ────────────────────────────────
        _cls_i = (track_i.current_detection.get('class_name') or '').lower()
        _cls_j = (track_j.current_detection.get('class_name') or '').lower()
        if _cls_i in ('truck', 'bus') or _cls_j in ('truck', 'bus'):
            return None

        pair_key = (min(track_i.track_id, track_j.track_id),
                    max(track_i.track_id, track_j.track_id))

        # ── Hitung keempat primitif ─────────────────────────────────────────
        ce_raw              = self._compute_CE(track_i, track_j, frame_id)
        kelr_i, kelr_val_i  = self._compute_KELR_personal(track_i)
        kelr_j, kelr_val_j  = self._compute_KELR_personal(track_j)
        kelr                = kelr_i or kelr_j
        ars_i,  ars_z_i     = self._compute_ARS(track_i)
        ars_j,  ars_z_j     = self._compute_ARS(track_j)
        ars                 = ars_i or ars_j
        pmv,    pmv_val     = self._compute_PMV(track_i, track_j)

        # Q8 primitive ablation mask. Values are zeroed before any path state is
        # updated so disabled primitives cannot contribute through rolling state.
        if not self.enable_primitive_ce:
            ce_raw = False
        if not self.enable_primitive_kelr:
            kelr_i = kelr_j = kelr = False
            kelr_val_i = kelr_val_j = 0.0
        if not self.enable_primitive_ars:
            ars_i = ars_j = ars = False
            ars_z_i = ars_z_j = 0.0
        if not self.enable_primitive_pmv:
            pmv = False
            pmv_val = 0.0

        # iou_now untuk KELR proximity guard dan CE confirmation
        bbox_i  = track_i.current_detection.get('bbox')
        bbox_j  = track_j.current_detection.get('bbox')
        iou_now = self._iou(bbox_i, bbox_j) if (bbox_i is not None and bbox_j is not None) else 0.0

        # ── Rolling KELR/ARS recent max (selalu diupdate, decay per frame) ───────
        # Dipakai untuk: (a) Path-D pre-streak lookback, (b) post-CE inheritance
        _decay = self.PATH_D_RECENT_DECAY
        self._pair_kelr_recent[pair_key] = max(
            self._pair_kelr_recent.get(pair_key, -999.0) * _decay,
            max(kelr_val_i, kelr_val_j)
        )
        self._pair_ars_recent[pair_key] = max(
            self._pair_ars_recent.get(pair_key, 0.0) * _decay,
            max(ars_z_i, ars_z_j)
        )

        # ── Path D streak tracking + kinematic evidence accumulation ───────────
        if iou_now >= self.IOU_PERSIST_MIN:
            self._iou_contact_streak[pair_key] = \
                self._iou_contact_streak.get(pair_key, 0) + 1
            # Akumulasi max KELR dan ARS selama streak berlangsung
            self._iou_streak_kelr_max[pair_key] = max(
                self._iou_streak_kelr_max.get(pair_key, -999.0),
                max(kelr_val_i, kelr_val_j)
            )
            self._iou_streak_ars_max[pair_key] = max(
                self._iou_streak_ars_max.get(pair_key, 0.0),
                max(ars_z_i, ars_z_j)
            )
        else:
            self._iou_contact_streak[pair_key] = 0
            self._iou_streak_kelr_max.pop(pair_key, None)
            self._iou_streak_ars_max.pop(pair_key, None)
        _persist_count = self._iou_contact_streak.get(pair_key, 0)

        # ── Path E state update ───────────────────────────────────────────────
        # CE_raw proximity TTL — set/refresh saat ce_raw, hitung mundur setiap frame.
        if ce_raw:
            self._path_e_ce_recent[pair_key] = self.PATH_E_CE_WINDOW
        elif pair_key in self._path_e_ce_recent:
            self._path_e_ce_recent[pair_key] -= 1
            if self._path_e_ce_recent[pair_key] <= 0:
                del self._path_e_ce_recent[pair_key]
        # Burst counter: naik jika KELR+ARS keduanya di atas threshold, reset jika tidak.
        _e_kelr_max = max(kelr_val_i, kelr_val_j)
        _e_ars_max  = max(ars_z_i, ars_z_j)
        if _e_kelr_max > self.PATH_E_KELR_MIN and _e_ars_max > self.PATH_E_ARS_MIN:
            self._path_e_burst[pair_key] = self._path_e_burst.get(pair_key, 0) + 1
        else:
            self._path_e_burst[pair_key] = 0

        # ── CE Confirmation: deepening contact (Blueprint A7) ────────────────
        #
        # Pelajaran dari iterasi sebelumnya:
        #   A4: `ce or (kelr and iou>0)` — CE standalone → FP papasan ❌
        #   A5: `kelr and (ce or iou>0)` — KELR wajib → FN sideswipe ❌
        #   A6: `(ce and (kelr or ars)) or (kelr and iou>0)`
        #       — CE + ARS → FP dari belok/proximity (ARS satu sisi) ❌
        #
        # Root cause A6: CE fires pada kontak PERTAMA (IoU 0→positif).
        # Pada saat itu, ARS bisa sudah fire hanya untuk SATU kendaraan
        # (misalnya yang belok atau yang baru saja menyentuh). Ini menyebabkan
        # FP dari: belok kanan/kiri, kendaraan terlalu dekat, papasan sekilas.
        #
        # Fix A7: CE_confirmed hanya jika IoU MENINGKAT di frame berikutnya.
        #   - Tabrakan nyata: kontak semakin dalam (IoU naik frame N+1) ✓
        #   - Papasan/belok/proximity: IoU kembali ke 0 → tidak dikonfirmasi ✓
        #
        # Implementasi tiga-langkah (dua kenaikan IoU berturut-turut):
        #   Frame N   : ce_raw=True → pending[pair] = {iou=X, stage=0}, ce=False
        #   Frame N+1 : stage=0
        #     - iou_now > X → stage→1, rise_iou=iou_now (first rise dicatat)
        #     - iou_now ≤ X → grazing (papasan/belok), TTL counting
        #   Frame N+2 : stage=1
        #     - iou_now > rise_iou → CE CONFIRMED (second rise, deepening nyata) ✓
        #     - iou_now ≤ rise_iou → kontak tidak semakin dalam → discard ✗
        #   TTL expired di stage manapun → discard ✗
        #
        # Crossing sekilas (intersection):  IoU naik 1 frame lalu drop → stage 1 tapi
        # tidak ada second rise → DISCARD.  Papasan anti-parallel → A8.2 suppress.
        # Tabrakan nyata: IoU naik ≥ 2 kali berturut-turut → CONFIRMED.
        if ce_raw:
            # Kontak pertama — simpan sebagai pending dengan evidence saat ini
            # (overwrite jika pair sudah pending, artinya kontak baru setelah separasi)
            self._pending_ce[pair_key] = {
                'iou':  iou_now,    # baseline IoU di frame CE_raw (harus dilewati = 1 rise)
                'kelr': kelr,
                'ars':  ars,
                'ttl':  self.CE_CONFIRM_WINDOW,
            }
            # Catat frame terakhir CE_raw fire untuk Path A Sliding Window
            self._ce_raw_last_frame[pair_key] = frame_id
            ce = False
        elif pair_key in self._pending_ce:
            pending = self._pending_ce[pair_key]
            # Accumulate evidence selama pending window (KELR/ARS dari frame manapun diwariskan)
            pending['kelr'] = pending['kelr'] or kelr
            pending['ars']  = pending['ars']  or ars
            pending['ttl'] -= 1

            # ── REDESAIN: 1-stage confirmation (bukan 2-stage) ───────────────
            # Data empiris: 2-stage (2 consecutive IoU rises) membunuh 88% sinyal valid
            # karena post-collision IoU plateau atau turun (separasi setelah impact).
            # Solusi: 1 rise saja sudah cukup membuktikan deepening contact.
            # Papasan tetap aman karena IoU tidak naik sama sekali (turun ke 0).
            if iou_now > pending['iou'] and iou_now > 0:
                # Rise pertama → CE CONFIRMED (kontak semakin dalam)
                # Wariskan semua evidence yang terkumpul sejak CE_raw
                ce   = True
                kelr = kelr or pending['kelr']
                ars  = ars  or pending['ars']
                del self._pending_ce[pair_key]
                # Post-CE window: beri KELR/ARS waktu tambahan untuk fire
                self._post_ce_window[pair_key] = self.POST_CE_WINDOW
            elif pending['ttl'] <= 0:
                # Window habis tanpa rise → papasan/belok sekilas → buang
                del self._pending_ce[pair_key]
                ce = False
            else:
                ce = False          # masih menunggu dalam window
        else:
            ce = False

        # ── Logic Blueprint A8 ───────────────────────────────────────────────
        #
        # KELR proximity guard: menggantikan `iou_now > 0` dengan distance-based check.
        # Alasan: di kamera overhead, IoU bbox selalu 0 meskipun kendaraan bersentuhan.
        # Pengganti scale-free: center_dist < 2.0 × max_bbox_diagonal
        # Fisika: KELR mendeteksi "kendaraan mendadak berhenti". Tanpa proximity guard,
        # KELR akan dipasangkan dengan semua track di scene → FP masif.
        # 2.0× memberi ruang lebih dari CE (1.5×) untuk KELR yang tidak mensyaratkan
        # arah mendekati (bisa rear-end setelah kontak).
        _camera_angle = str(self._scalar_log_angle or "unknown").lower()
        _scene_cat = str(self._scene_category or "unknown").lower()
        _strict_scene = (
            _camera_angle in ("overhead", "nadir", "topdown", "top-down") or
            _scene_cat in ("dense", "congested")
        )
        KELR_PROX_DIAG_FACTOR = self.KAPPA_PROX_STRICT if _strict_scene else self.KAPPA_PROX
        _bbox_i = track_i.current_detection.get('bbox', [0, 0, 1, 1])
        _bbox_j = track_j.current_detection.get('bbox', [0, 0, 1, 1])
        _diag_i = math.sqrt(max(_bbox_i[2] - _bbox_i[0], 1) ** 2 + max(_bbox_i[3] - _bbox_i[1], 1) ** 2)
        _diag_j = math.sqrt(max(_bbox_j[2] - _bbox_j[0], 1) ** 2 + max(_bbox_j[3] - _bbox_j[1], 1) ** 2)
        _max_diag = max(_diag_i, _diag_j, 1.0)
        _ci = track_i.current_detection.get('center')
        _cj = track_j.current_detection.get('center')
        if _ci is None:
            _ci = [(_bbox_i[0] + _bbox_i[2]) / 2.0, (_bbox_i[1] + _bbox_i[3]) / 2.0]
        if _cj is None:
            _cj = [(_bbox_j[0] + _bbox_j[2]) / 2.0, (_bbox_j[1] + _bbox_j[3]) / 2.0]
        _center_dist = math.sqrt((_ci[0] - _cj[0]) ** 2 + (_ci[1] - _cj[1]) ** 2)
        _in_proximity = _center_dist < _max_diag * KELR_PROX_DIAG_FACTOR

        # ── Direction guard untuk mutual KELR (IoU = 0 only) ─────────────────
        #
        # Problem: papasan + coincidental braking.
        # Dua mobil dari arah berlawanan yang sama-sama ngerem (misalnya:
        #   Track 1 berhenti karena tertabrak truk,
        #   Track 2 berhenti karena ada kecelakaan di depannya)
        # bisa trigger mutual KELR meski tidak saling bertabrakan.
        # IoU = 0 karena bbox tidak tumpang tindih, tapi proximity masih terpenuhi.
        #
        # Fix: jika IoU = 0 (tidak ada kontak bbox), terapkan direction guard:
        #   cos(net_disp_i, net_disp_j) < -0.5 → angle > 120° → berlawanan arah → suppress.
        #
        # Fisika:
        #   Tabrakan head-on nyata (IoU = 0) → ditangkap oleh CE path via prox_contact.
        #   Overhead rear-end (IoU = 0, arah sama) → cos ≥ -0.5 → lolos.
        #   Papasan + braking independen (arah berlawanan) → cos < -0.5 → ditolak. ✓
        #
        # Guard TIDAK berlaku jika IoU > 0 — kontak bbox sudah membuktikan kedekatan fisik.
        # (Identik filosofinya dengan A8.2, hanya path yang berbeda.)
        _kelr_mutual_dir_ok = True
        if kelr_i and kelr_j and _in_proximity and iou_now == 0.0:
            _ndi = self._net_displacement_vector(track_i)
            _ndj = self._net_displacement_vector(track_j)
            _si  = float(np.linalg.norm(_ndi))
            _sj  = float(np.linalg.norm(_ndj))
            if _si > 1e-6 and _sj > 1e-6:
                _cos_kd = float(np.dot(_ndi, _ndj)) / (_si * _sj)
                if _cos_kd < -0.5:   # angle > 120°: berlawanan arah → suppress
                    _kelr_mutual_dir_ok = False
            # Guard: kedua kendaraan berhenti penuh (KELR_val > 0.85) + no contact (IoU=0).
            # Suppress HANYA jika arah gerak sama (queue/convoy braking, cos > 0.5).
            # T-bone (cos ≈ 0) dan head-on (cos < -0.5, sudah di-suppress di atas)
            # adalah collision nyata — jangan suppress.
            if kelr_val_i > 0.85 and kelr_val_j > 0.85:
                if _si > 1e-6 and _sj > 1e-6 and _cos_kd > 0.5:
                    _kelr_mutual_dir_ok = False  # same-direction queue braking → suppress
                elif not (_si > 1e-6 and _sj > 1e-6):
                    _kelr_mutual_dir_ok = False  # no movement history → suppress conservatively

            # ── Exception: mutual physics-threshold KELR + very tight proximity ──
            # Fisika: dua kendaraan DALAM JARAK SATU DIAGONAL saling berhenti tiba-tiba
            # adalah bukti kuat tabrakan meskipun arah berlawanan (kamera low-angle CCTV).
            # Di antrean merah: jarak antar kendaraan ≥ 1 panjang mobil ≈ 1 diagonal;
            # IoU=0 berarti tidak ada tumpang tindih bbox → center_dist > diagonal minimal.
            # Jika center_dist < 1.0× max_diag DAN kedua KELR ≥ 0.91 (threshold fisika Brach
            # 1991, bukan tuning), ini adalah kolisi — bukan pengereman independen.
            # Guard direction sudah memastikan anti-parallel (head-on collision);
            # exception ini khusus untuk kasus di mana kamera low-angle menyebabkan IoU=0
            # meskipun secara fisik kendaraan sudah bersentuhan.
            if (kelr_val_i >= self.KELR_THRESHOLD and
                    kelr_val_j >= self.KELR_THRESHOLD and
                    _center_dist < _max_diag * 1.0):
                _kelr_mutual_dir_ok = True   # Override — tight proximity + dual physics-threshold KELR

        # ── Path C: KELR + kontak langsung (IoU > 0) ─────────────────────────
        #
        # Fisika: jika satu track mendadak berhenti (KELR) DAN bbox-nya tumpang
        # tindih dengan track lain (IoU > 0), maka ada kontak fisik langsung.
        # Ini lebih kuat dari Path B (mutual KELR tanpa kontak) karena IoU > 0
        # sudah membuktikan kedekatan secara geometris.
        #
        # Kasus yang tertangkap tapi tidak bisa oleh Path A atau B:
        #   - Korban (Track 1) berhenti tiba-tiba karena ditabrak (KELR fires)
        #   - Pelaku (Track 4) masih bergerak (tidak punya KELR)
        #   - Bbox korban & pelaku overlap (IoU = 0.074)
        #   → Path A: CE mungkin sudah expired sebelum KELR sync  [timing mismatch]
        #   → Path B: hanya Track 1 punya KELR, bukan mutual  [tidak fire]
        #   → Path C: kelr_i AND iou > 0  →  TRIGGERED  ✓
        #
        # Direction guard untuk single-KELR:
        #   Jika hanya satu track punya KELR (asimetrik) DAN arah berlawanan
        #   (papasan, salah satu ngerem independen) → SUPPRESS.
        #   Mutual KELR + IoU > 0 selalu OK tanpa direction guard
        #   (dua kendaraan saling berhenti sambil bersentuhan = collision).
        _kelr_iou_ok = (kelr_i or kelr_j) and iou_now > 0
        if _kelr_iou_ok:
            # ── IoU delta guard ──────────────────────────────────────────────
            # Tabrakan nyata: IoU melompat mendadak dalam 1-2 frame (kontak fisik tiba-tiba).
            # Perspektif / overtaking: IoU tumbuh perlahan selama puluhan frame karena
            #   dua kendaraan bergerak paralel di kamera sempit → bukan kontak fisik.
            #
            # Threshold 0.02/frame:
            #   Perspektif overtaking: IoU delta ≈ 0.001-0.005/frame  →  di bawah threshold
            #   Tabrakan nyata (rear-end, T-bone): delta ≈ 0.05-0.5/frame  →  di atas threshold
            _iou_hist_c = self._iou_history.get(pair_key, [iou_now])
            _iou_prev_c = _iou_hist_c[-2] if len(_iou_hist_c) >= 2 else 0.0
            _KELR_IOU_DELTA_MIN = self.KELR_IOU_DELTA_MIN
            if (iou_now - _iou_prev_c) < _KELR_IOU_DELTA_MIN:
                # Exception: KELR sudah di physics-threshold (0.91) → kendaraan full-stop.
                # Post-impact IoU tidak akan naik lagi (tabrakan sudah selesai) tapi collision nyata.
                # FP overtaking/perspektif punya KELR sub-threshold (0.3–0.7) → tidak lolos exception ini.
                _kelr_at_threshold = max(kelr_val_i, kelr_val_j) >= self.KELR_THRESHOLD
                if not _kelr_at_threshold:
                    _kelr_iou_ok = False   # IoU tumbuh lambat + KELR sub-threshold = perspektif/paralel

        if _kelr_iou_ok and not (kelr_i and kelr_j):
            # Single-KELR: terapkan direction guard
            # Fix 1: skip direction guard jika history track pendek (< W/2).
            # Di kamera low-angle CCTV, displacement vector tidak reliable untuk
            # track yang baru muncul — collision bisa terjadi kapan saja.
            #
            # Fix 2: skip direction guard jika track yang KELR-nya fire juga fire ARS
            # sekaligus (dual kinematic evidence pada satu kendaraan).
            # Fisika: KELR + ARS bersamaan = mendadak berhenti DAN mengalami
            # deformasi geometris — kombinasi yang hampir eksklusif untuk tabrakan nyata.
            # Direction guard tidak reliable di kamera low-angle untuk skenario ini;
            # kekeliruan arah lebih mungkin terjadi daripada FP dari dual kinematic.
            _dual_kinematic_c = (kelr_i and ars_i) or (kelr_j and ars_j)

            # Fix 3: skip direction guard jika KELR sudah di physics-threshold (0.91).
            # Papasan: KELR sub-threshold (ngerem biasa, 0.3–0.7) → direction guard tetap berlaku.
            # Head-on collision: KELR ≥ 0.91 (full-stop dari impact) → anti-parallel adalah
            # signature head-on nyata, bukan papasan. Direction guard salah konteks di sini.
            _kelr_physics_threshold = max(kelr_val_i, kelr_val_j) >= self.KELR_THRESHOLD

            if not _dual_kinematic_c and not _kelr_physics_threshold:
                # Tidak ada dual kinematic evidence — terapkan direction guard normal
                _hist_c_i = len(self._get_raw_centers(track_i, last_n=self.W))
                _hist_c_j = len(self._get_raw_centers(track_j, last_n=self.W))
                if _hist_c_i >= self.W // 2 and _hist_c_j >= self.W // 2:
                    _ndi2 = self._net_displacement_vector(track_i)
                    _ndj2 = self._net_displacement_vector(track_j)
                    _si2  = float(np.linalg.norm(_ndi2))
                    _sj2  = float(np.linalg.norm(_ndj2))
                    if _si2 > 1e-6 and _sj2 > 1e-6:
                        _cos_c = float(np.dot(_ndi2, _ndj2)) / (_si2 * _sj2)
                        if _cos_c < -0.5:   # angle > 120°: berlawanan arah → papasan → suppress
                            _kelr_iou_ok = False

        # Strict overhead/dense guard:
        # Single-track KELR is the dominant FP mode in dense/intersection logs.
        # In this operating regime, KELR is supporting evidence, not a final
        # trigger unless contact changes abruptly and a second impact cue exists.
        # IoU > 0.90 is usually duplicate-track overlap, not two physical objects.
        if _kelr_iou_ok and iou_now > 0.90:
            _kelr_iou_ok = False

        if _strict_scene and _kelr_iou_ok:
            _iou_hist_s = self._iou_history.get(pair_key, [iou_now])
            _iou_prev_s = _iou_hist_s[-2] if len(_iou_hist_s) >= 2 else 0.0
            _iou_delta_s = iou_now - _iou_prev_s
            _single_kelr_s = (kelr_i != kelr_j)
            _same_track_dual_s = (kelr_i and ars_i) or (kelr_j and ars_j)
            _contact_impulse_s = ce or _same_track_dual_s

            if _single_kelr_s:
                _kelr_iou_ok = (
                    iou_now >= 0.05 and
                    _iou_delta_s >= self.STRICT_SINGLE_KELR_IOU_DELTA_MIN and
                    _contact_impulse_s
                )
            else:
                _kelr_iou_ok = (
                    iou_now >= 0.05 and
                    _iou_delta_s >= -0.005
                )

        # Blueprint A8: triggered jika:
        #   (a) CE terkonfirmasi + setidaknya satu primitive kinetik (KELR/ARS), ATAU
        #   (b) KELR MUTUAL (kedua track mendadak decelerate) + proximity + direction guard.
        #       BUKAN single-track KELR — di kamera overhead, ripple effect collision
        #       menyebabkan kendaraan lain di sekitar juga decelerate, sehingga
        #       single-KELR + proximity menghasilkan FP masif.
        #       Mengharuskan KELR pada KEDUA track (kelr_i AND kelr_j) jauh lebih
        #       spesifik: hanya pasangan yang sama-sama berhenti tiba-tiba yang lolos.
        #       Direction guard tambahan: suppress mutual KELR jika arah berlawanan + IoU=0.
        #   (c) KELR (satu atau keduanya) + kontak langsung (IoU > 0).
        #       Path C menangani kasus timing mismatch CE+KELR: korban berhenti (KELR)
        #       saat bbox-nya tumpang tindih dengan pelaku (IoU > 0).
        #       Direction guard: single-KELR + arah berlawanan = papasan → suppress.
        # ── Path D: Persistent IoU contact (slow-speed collision) ─────────────
        # Fisika: jika dua kendaraan mempertahankan bbox overlap selama ≥ N frame
        # DAN keduanya tadinya bergerak, itu adalah bukti kontak fisik.
        # KELR tidak diperlukan — cocok untuk low-speed collision di persimpangan
        # di mana energy drop terlalu kecil untuk terdeteksi.
        #
        # Guards:
        #   (1) was_moving: kedua track harus punya net_disp > bbox_diag/4
        #       (mencegah FP dari kendaraan parkir yang saling berdekatan)
        #   (2) direction guard: sama seperti Path B/C
        #       (mencegah FP dari antiparallel passing)
        _path_d_ok = False
        # Guard: IoU > 0.9 = track duplication (ByteTrack assign 2 ID ke 1 objek).
        # Dua objek fisik berbeda tidak mungkin IoU > 0.9 — skip Path D.
        if iou_now > 0.9:
            self._iou_contact_streak[pair_key] = 0  # reset streak agar tidak carry over
        if _persist_count >= self.IOU_PERSIST_N and iou_now <= 0.9:
            _mob_i = self._net_displacement_in_window(track_i)
            _mob_j = self._net_displacement_in_window(track_j)
            _diag_i = math.sqrt(max(bbox_i[2]-bbox_i[0], 1)**2 +
                                 max(bbox_i[3]-bbox_i[1], 1)**2) if bbox_i else 1.0
            _diag_j = math.sqrt(max(bbox_j[2]-bbox_j[0], 1)**2 +
                                 max(bbox_j[3]-bbox_j[1], 1)**2) if bbox_j else 1.0
            # was_moving: net_disp > diag/4 ATAU ada speed riwayat > 1 px/fr
            # (kendaraan yang sudah berhenti setelah tabrakan tetap lolos jika sempat bergerak)
            def _ever_moved(trk, min_speed=1.0):
                # Cek apakah track BARU-BARU INI bergerak (IOU_PERSIST_N*2 frame terakhir).
                # Sengaja TIDAK cek seluruh W frame — kendaraan yang sudah lama berhenti
                # di persimpangan juga punya history gerak tapi bukan post-collision.
                recent_n = self.IOU_PERSIST_N * 2   # = 10 frame
                speeds = []
                centers = self._get_raw_centers(trk, last_n=recent_n)
                for k in range(1, len(centers)):
                    v = float(np.linalg.norm(
                        np.array(centers[k]) - np.array(centers[k-1])))
                    speeds.append(v)
                return max(speeds, default=0.0) > min_speed
            # _ever_moved hanya sebagai fallback jika net_disp > diag/12 (minimum gerak nyata).
            # Angka 1/12 dipilih empiris: cukup besar untuk reject kendaraan yang
            # hampir diam di kemacetan (net_disp << diag), tapi cukup kecil untuk
            # tetap menerima kendaraan post-collision yang sudah berhenti setelah bergerak.
            _mob_ok_i = (_mob_i > _diag_i / 4.0) or \
                        (_mob_i > _diag_i / 12.0 and _ever_moved(track_i, min_speed=3.0))
            _mob_ok_j = (_mob_j > _diag_j / 4.0) or \
                        (_mob_j > _diag_j / 12.0 and _ever_moved(track_j, min_speed=3.0))
            _both_moving = _mob_ok_i and _mob_ok_j
            import logging as _log; _log.getLogger(__name__).debug(
                f"[PATH-D] pair=({track_i.track_id},{track_j.track_id}) "
                f"streak={_persist_count} mob_i={_mob_i:.1f}/{_diag_i/4:.1f}(d/8:{_diag_i/8:.1f}) "
                f"mob_j={_mob_j:.1f}/{_diag_j/4:.1f}(d/8:{_diag_j/8:.1f}) "
                f"ever_i={_ever_moved(track_i, min_speed=3.0)} ever_j={_ever_moved(track_j, min_speed=3.0)} "
                f"both_moving={_both_moving}"
            )
            if _both_moving:
                # Direction guard: suppress antiparallel-passing FP (vehicles crossing paths).
                # BUT skip guard if track history is short (< W/2 frames): early-collision
                # tracks have unreliable displacement vectors (impact already changed direction).
                # Papasan hanya terjadi di tracks dengan riwayat panjang; collision bisa terjadi
                # kapan saja termasuk frame awal video.
                _hist_i = len(self._get_raw_centers(track_i, last_n=self.W))
                _hist_j = len(self._get_raw_centers(track_j, last_n=self.W))
                _enough_history = (_hist_i >= self.W // 2) and (_hist_j >= self.W // 2)

                _dir_ok_d = True
                if _enough_history:
                    _nd_i = self._net_displacement_vector(track_i)
                    _nd_j = self._net_displacement_vector(track_j)
                    _si_d = float(np.linalg.norm(_nd_i))
                    _sj_d = float(np.linalg.norm(_nd_j))
                    if _si_d > 1e-6 and _sj_d > 1e-6:
                        _cos_d = float(np.dot(_nd_i, _nd_j)) / (_si_d * _sj_d)
                        if _cos_d < -0.5:   # antiparallel → papasan, bukan tabrakan
                            _dir_ok_d = False
                if _dir_ok_d:
                    # Guard: jika kedua KELR value negatif, kedua kendaraan akselerasi.
                    # Tabrakan selalu menyebabkan setidaknya satu kendaraan melambat
                    # (KELR_val ≥ 0).  Kedua nilai negatif = crossing/overtaking di
                    # persimpangan padat, bukan kontak fisik.
                    if not (kelr_val_i < 0 and kelr_val_j < 0):
                        # ── Kinematic guard (redesign) ──────────────────────────
                        # Syarat baru: selama streak IoU ini, harus ada setidaknya
                        # satu sinyal energi (KELR atau ARS) meskipun sub-threshold.
                        # Ini membedakan kontak fisik (ada deceleration/spin)
                        # dari overlap perspektif / berhenti di lampu merah.
                        # Pre-streak + during-streak kinematic evidence
                        # (rolling recent menangkap KELR yang fire sebelum IoU streak)
                        _streak_kelr = max(
                            self._iou_streak_kelr_max.get(pair_key, -999.0),
                            self._pair_kelr_recent.get(pair_key, -999.0)
                        )
                        _streak_ars = max(
                            self._iou_streak_ars_max.get(pair_key, 0.0),
                            self._pair_ars_recent.get(pair_key, 0.0)
                        )
                        _kinematic_in_streak = (
                            _streak_kelr > self.PATH_D_KELR_MIN or
                            _streak_ars  > self.PATH_D_ARS_MIN
                        )
                        if _kinematic_in_streak:
                            _path_d_ok = True

        # ── [DISABLED] Path D dinonaktifkan sementara untuk eksperimen ──────────
        # Tujuan: verifikasi apakah 62.1% FP inter3 + 1.7% highway2 berasal dari D.
        # Re-enable: hapus baris berikut dan uncomment Path D di triggered.
        _path_d_ok = False  # DISABLED

        # ── Path E: Dual Kinematic Burst (safety net) ────────────────────────
        _path_e_ok = False
        _e_burst_count = self._path_e_burst.get(pair_key, 0)
        _e_ce_seen     = pair_key in self._path_e_ce_recent
        if _e_burst_count >= self.PATH_E_BURST_N and _e_ce_seen:
            # Direction guard — sama seperti Path D (skip jika history pendek)
            _dir_ok_e = True
            _hist_i_e = len(self._get_raw_centers(track_i, last_n=self.W))
            _hist_j_e = len(self._get_raw_centers(track_j, last_n=self.W))
            if _hist_i_e >= self.W // 2 and _hist_j_e >= self.W // 2:
                _nd_i_e = self._net_displacement_vector(track_i)
                _nd_j_e = self._net_displacement_vector(track_j)
                _si_e   = float(np.linalg.norm(_nd_i_e))
                _sj_e   = float(np.linalg.norm(_nd_j_e))
                if _si_e > 1e-6 and _sj_e > 1e-6:
                    _cos_e = float(np.dot(_nd_i_e, _nd_j_e)) / (_si_e * _sj_e)
                    if _cos_e < -0.5:    # antiparallel → papasan, bukan tabrakan
                        _dir_ok_e = False
            # Both-moving guard
            _mob_i_e  = self._net_displacement_in_window(track_i)
            _mob_j_e  = self._net_displacement_in_window(track_j)
            _diag_i_e = math.sqrt(max(bbox_i[2]-bbox_i[0], 1)**2 +
                                   max(bbox_i[3]-bbox_i[1], 1)**2) if bbox_i else 1.0
            _diag_j_e = math.sqrt(max(bbox_j[2]-bbox_j[0], 1)**2 +
                                   max(bbox_j[3]-bbox_j[1], 1)**2) if bbox_j else 1.0
            _mob_ok_e = (_mob_i_e > _diag_i_e / 4.0) and (_mob_j_e > _diag_j_e / 4.0)
            if _dir_ok_e and _mob_ok_e:
                _path_e_ok = True

        # ── [DISABLED] Path E dinonaktifkan sementara untuk eksperimen ──────────
        # Tujuan: verifikasi apakah 8.2% FP highway + 37.9% FP inter3 berasal dari E.
        # Re-enable: hapus baris berikut dan uncomment Path E di triggered.
        _path_e_ok = False  # DISABLED

        # ── ARS guard untuk CE-based paths (Fix 2a/2b) ──────────────────────
        # Di kamera oblique highway, ARS 2.0-3.0σ mudah terpenuhi dari perubahan
        # perspektif saat kendaraan bergerak — bukan deformasi tabrakan nyata.
        # Jika KELR tidak fire (speed drop tidak terdeteksi), ARS hanya diterima jika:
        #   (a) IoU >= PATH_A_SW_MIN_IOU (ada overlap bbox fisik nyata), DAN
        #   (b) ARS >= ARS_CE_SIGMA_HIGH (ekstrem, bukan sekadar perspektif)
        # Jika KELR fire, ARS diterima tanpa syarat tambahan (KELR sudah jadi anchor).
        _ars_z_max = max(ars_z_i, ars_z_j)
        _ars_strong = ars and (
            kelr or
            (iou_now >= self.PATH_A_SW_MIN_IOU and _ars_z_max >= self.ARS_CE_SIGMA_HIGH)
        )

        # ── Post-CE kinematic window (Fix V7: KELR fire setelah CE confirmed) ──
        # Setelah CE confirmed, beri KELR/ARS POST_CE_WINDOW frame untuk fire.
        # Menangkap kasus di mana velocity drop teramati beberapa frame setelah
        # kontak geometrik terkonfirmasi (misalnya: impact → kendaraan berhenti 3 frame kemudian).
        _post_ce_kinematic = False
        if pair_key in self._post_ce_window:
            if kelr or _ars_strong:
                _post_ce_kinematic = True
                del self._post_ce_window[pair_key]
            else:
                self._post_ce_window[pair_key] -= 1
                if self._post_ce_window[pair_key] <= 0:
                    del self._post_ce_window[pair_key]

        # ── Path A Sliding Window (REDESAIN) ────────────────────────────────
        # CE_raw dan KELR tidak harus co-occur dalam 1 frame.
        # Cukup keduanya pernah fire dalam PATH_A_WINDOW frame terakhir.
        # Guard: minimal ada proximity (IoU > 0 atau pair sedang dalam monitoring)
        _path_a_sw_ok = False
        if kelr_i or kelr_j:
            # Update frame terakhir KELR fire
            self._kelr_last_frame[pair_key] = frame_id
        _last_ce_raw  = self._ce_raw_last_frame.get(pair_key, -9999)
        _last_kelr    = self._kelr_last_frame.get(pair_key, -9999)
        if (_last_ce_raw >= 0 and _last_kelr >= 0 and
                abs(frame_id - _last_ce_raw)  <= self.PATH_A_WINDOW and
                abs(frame_id - _last_kelr)    <= self.PATH_A_WINDOW and
                (kelr_i or kelr_j) and
                iou_now >= self.PATH_A_SW_MIN_IOU):   # harus ada overlap bbox nyata (bukan sekadar center proximity)
            # Direction guard: sama dengan Path C/D — suppress antiparallel passing
            _sw_dir_ok = True
            _hist_sw_i = len(self._get_raw_centers(track_i, last_n=self.W))
            _hist_sw_j = len(self._get_raw_centers(track_j, last_n=self.W))
            if _hist_sw_i >= self.W // 2 and _hist_sw_j >= self.W // 2:
                _nd_sw_i = self._net_displacement_vector(track_i)
                _nd_sw_j = self._net_displacement_vector(track_j)
                _si_sw   = float(np.linalg.norm(_nd_sw_i))
                _sj_sw   = float(np.linalg.norm(_nd_sw_j))
                if _si_sw > 1e-6 and _sj_sw > 1e-6:
                    _cos_sw = float(np.dot(_nd_sw_i, _nd_sw_j)) / (_si_sw * _sj_sw)
                    if _cos_sw < -0.5:   # antiparallel = papasan → suppress
                        _sw_dir_ok = False
            if _sw_dir_ok:
                _path_a_sw_ok = True

        _pmv_ce_gated = False  # DISABLED — masih spam FP, butuh investigasi lebih lanjut

        _path_a_ok = ce and (kelr or _ars_strong)
        _path_b_ok = kelr_i and kelr_j and _in_proximity and _kelr_mutual_dir_ok
        _path_c_ok = _kelr_iou_ok

        if _strict_scene:
            # In dense/top-down scenes, CE+ARS-only and delayed ARS-only are common
            # normal-turn artefacts. Keep CE paths only when KELR anchors the event.
            _path_a_ok = ce and kelr
            if _post_ce_kinematic and not kelr:
                _post_ce_kinematic = False

            # Mutual KELR without tight contact is usually stop-and-go ripple.
            _path_b_ok = _path_b_ok and (
                iou_now >= 0.05 or
                _center_dist < _max_diag * 1.0
            )

        triggered = (
            (ce and (kelr or _ars_strong)) or                                # Path A: CE + kinetik (ARS-only: IoU>=0.05 + ARS>=3σ)
            _post_ce_kinematic or                                            # Path A+: CE confirmed + KELR/ARS delayed
            _path_a_sw_ok or                                                 # Path A_sw: CE_raw+KELR dalam sliding window
            (kelr_i and kelr_j and _in_proximity and _kelr_mutual_dir_ok) or # Path B: mutual KELR
            _kelr_iou_ok or                                                  # Path C: KELR + kontak
            _pmv_ce_gated                                                    # Path PMV: transfer momentum impulsif, di-gate ke CE_raw
            # _path_d_ok or   # [DISABLED] Path D — dinonaktifkan sementara
            # _path_e_ok      # [DISABLED] Path E — dinonaktifkan sementara
        )

        # Override the legacy inline expression above with path variables so
        # strict-scene suppression is applied consistently.
        triggered = (
            _path_a_ok or
            _post_ce_kinematic or
            _path_a_sw_ok or
            _path_b_ok or
            _path_c_ok or
            _pmv_ce_gated
        )

        # ── Scalar logging (discriminability evaluation) ─────────────────────
        # Dicatat SEBELUM early-return agar nilai non-triggered pair juga tersimpan.
        # Ini yang memungkinkan ROC/AUC per-primitive di analisis offline.
        # kelr_max / ars_z_max = nilai tertinggi dari kedua track (lebih informatif
        # untuk threshold sweep daripada nilai individual).

        # Inferensi path yang bertanggung jawab atas trigger (untuk analisis paper)
        # Priority: A > A+ > B > C > D > none
        if not triggered:
            _path_triggered = 'none'
        elif _path_a_ok:
            _path_triggered = 'A'
        elif _post_ce_kinematic:
            _path_triggered = 'A+'
        elif _path_a_sw_ok:
            _path_triggered = 'A_sw'
        elif _path_b_ok:
            _path_triggered = 'B'
        elif _path_c_ok:
            _path_triggered = 'C'
        elif _pmv_ce_gated:
            _path_triggered = 'PMV'
        elif _path_d_ok:
            _path_triggered = 'D'
        elif _path_e_ok:
            _path_triggered = 'E'
        else:
            _path_triggered = 'unknown'

        if self._scalar_log_enabled:
            # ── TTC / DRAC computation (for baseline comparison) ─────────────
            _bbox_i_l = track_i.current_detection.get('bbox', [0, 0, 1, 1])
            _bbox_j_l = track_j.current_detection.get('bbox', [0, 0, 1, 1])
            _ci_l = track_i.current_detection.get('center')
            _cj_l = track_j.current_detection.get('center')
            if _ci_l is None:
                _ci_l = [(_bbox_i_l[0]+_bbox_i_l[2])/2.0, (_bbox_i_l[1]+_bbox_i_l[3])/2.0]
            if _cj_l is None:
                _cj_l = [(_bbox_j_l[0]+_bbox_j_l[2])/2.0, (_bbox_j_l[1]+_bbox_j_l[3])/2.0]
            _dist_l   = math.sqrt((_ci_l[0]-_cj_l[0])**2 + (_ci_l[1]-_cj_l[1])**2)
            _vi_l     = self._raw_velocity(track_i)
            _vj_l     = self._raw_velocity(track_j)
            _spd_i_l  = float(np.linalg.norm(_vi_l))
            _spd_j_l  = float(np.linalg.norm(_vj_l))
            _d_vec_l  = np.array([_ci_l[0]-_cj_l[0], _ci_l[1]-_cj_l[1]])
            _d_norm_l = max(float(np.linalg.norm(_d_vec_l)), 1e-6)
            _d_unit_l = _d_vec_l / _d_norm_l
            _closing_l = max(-float(np.dot(_vi_l - _vj_l, _d_unit_l)), 0.0)
            _ttc_l  = round(_dist_l / _closing_l, 3) if _closing_l > 0.5 else 999.0
            _drac_l = round((_closing_l ** 2) / (2.0 * max(_dist_l, 1.0)), 4) if _closing_l > 0.5 else 0.0
            # ─────────────────────────────────────────────────────────────────
            self._scalar_log.append({
                'video':          self._scalar_log_video,
                'video_label':    self._scalar_log_label,
                'camera_angle':   self._scalar_log_angle,
                'frame':          frame_id,
                'track_i':        track_i.track_id,
                'track_j':        track_j.track_id,
                'kelr_i':         round(kelr_val_i, 4),
                'kelr_j':         round(kelr_val_j, 4),
                'kelr_max':       round(max(kelr_val_i, kelr_val_j), 4),
                'ars_z_i':        round(ars_z_i, 4),
                'ars_z_j':        round(ars_z_j, 4),
                'ars_z_max':      round(max(ars_z_i, ars_z_j), 4),
                'iou':            round(iou_now, 4),
                'ce_raw':         int(ce_raw),
                'ce_confirmed':   int(ce),
                'path_d':         int(_path_d_ok),
                'path_e':         int(_path_e_ok),
                'path_triggered': _path_triggered,
                'triggered':      int(triggered),
                'pmv_ratio':      round(pmv_val, 4),
                'scene_density':  round(self._scene_density, 2),
                'scene_category': self._scene_category,
                'strict_scene':   int(_strict_scene),
                'center_dist':    round(_dist_l, 2),
                'speed_i':        round(_spd_i_l, 4),
                'speed_j':        round(_spd_j_l, 4),
                'closing_speed':  round(_closing_l, 4),
                'ttc':            _ttc_l,
                'drac':           _drac_l,
            })
        # ─────────────────────────────────────────────────────────────────────

        if not triggered:
            return None

        # ── Guard: partner mobility (CE+ARS-only path) ──────────────────────
        #
        # Problem: kendaraan post-collision yang sudah DIAM di jalan dapat memicu
        # FP ketika kendaraan lain (bus, dll) lewat dekat:
        #   - Bus decelerate → ARS fires pada bus
        #   - Bus bbox overlap bbox wreck yang diam → CE deepens → trigger
        #
        # Fix: jika trigger via CE+ARS TANPA KELR, kedua track harus punya
        # net_displacement > bbox_diagonal/4 (sama dengan was_moving guard di ARS).
        #   - Tabrakan nyata: kedua kendaraan baru saja bergerak → mobile ✓
        #   - Bus lewat dekat wreck: wreck net_disp ≈ 0 → SUPPRESS ✓
        #
        # Guard TIDAK berlaku jika KELR fire (KELR sudah jadi gating yang kuat).
        if ce and _ars_strong and not kelr:
            # ── Sub-guard A8.1: Partner mobility ────────────────────────────
            # Kedua track harus punya net_displacement > bbox_diagonal/4.
            # Mencegah "bus lewat dekat post-collision wreck" FP.
            for chk_track, chk_bbox in ((track_i, bbox_i), (track_j, bbox_j)):
                if chk_bbox is None:
                    continue
                net_disp = self._net_displacement_in_window(chk_track)
                x1, y1, x2, y2 = chk_bbox
                bbox_diag = math.sqrt(max(x2 - x1, 1) ** 2 + max(y2 - y1, 1) ** 2)
                if net_disp <= bbox_diag / 4.0:
                    # Satu track stationary → post-collision wreck proximity → bukan tabrakan
                    return None

            # ── Sub-guard A8.2: Papasan filter (velocity angle) ─────────────
            # Papasan (kendaraan dari arah berlawanan) + CE + ARS tanpa KELR = FP.
            # Physics: tabrakan head-on NYATA → deceleration masif → KELR fires.
            # Jadi CE+ARS-only + arah berlawanan = papasan, BUKAN tabrakan.
            #
            # Implementasi: cos(θ) antara net displacement vectors (W-frame window).
            #   cos(θ) < -0.5 → angle > 120° → berlawanan arah → papasan → SUPPRESS.
            #   cos(θ) ≥ -0.5 → angle ≤ 120° → sama/tegak lurus → tabrakan → lanjut.
            #
            # Dimensionless: unit vector cosine, tidak ada pixel threshold.
            #
            # KENAPA net displacement, bukan instantaneous (_raw_velocity)?
            # _raw_velocity = history[-1] - history[-2] (1 frame).
            # Pada 1 frame jitter YOLO, kendaraan bergerak 10 px/fr bisa tampak
            # bergerak hanya 0.8 px (speed ≤ 1.0) → guard threshold dilewati → FP.
            # Net displacement atas W frame (≈1 detik) mererata jitter → arah reliable.
            # A8.1 sudah membuktikan kedua track punya net_disp > 0 (mobile),
            # jadi net displacement vector dijamin non-zero dan representatif.
            v_i_vec = self._net_displacement_vector(track_i)
            v_j_vec = self._net_displacement_vector(track_j)
            si = float(np.linalg.norm(v_i_vec))
            sj = float(np.linalg.norm(v_j_vec))
            if si > 1e-6 and sj > 1e-6:   # numerically nonzero (A8.1 garantees mobile)
                cos_angle = float(np.dot(v_i_vec, v_j_vec)) / (si * sj)
                # Relax: Papasan angle (< -0.5) di-bypass jika ada ARS extreme (> 4.0).
                # Tabrakan side-swipe/head-on nyata bisa menghasilkan rebound displacement aneh.
                ars_extreme = (ars_z_i > 4.0 or ars_z_j > 4.0)
                if cos_angle < -0.5 and not ars_extreme:
                    return None         # Papasan — bukan tabrakan

        primitives_fired = []
        if ce:          primitives_fired.append("CE")
        if kelr_i:      primitives_fired.append("KELR_personal_i")
        if kelr_j:      primitives_fired.append("KELR_personal_j")
        if ars:         primitives_fired.append(f"ARS({'i' if ars_i else 'j'})")
        if pmv:         primitives_fired.append(f"PMV({pmv_val:.2f})")
        if _path_d_ok:  primitives_fired.append(f"PERSIST({_persist_count}fr)")

        return {
            'track_id_1':       track_i.track_id,
            'track_id_2':       track_j.track_id,
            'frame_id':         frame_id,
            'primitives_fired': primitives_fired,
            'primitive_ce':     ce,
            'primitive_kelr':   kelr,
            'primitive_ars':    ars,
            'kelr_value_i':     round(kelr_val_i, 4),
            'kelr_value_j':     round(kelr_val_j, 4),
            'ars_zscore_i':     round(ars_z_i, 4),
            'ars_zscore_j':     round(ars_z_j, 4),
            'iou_now':          round(iou_now, 4),
            'path_triggered':   _path_triggered,
            'strict_scene':     _strict_scene,
            'source':           'PhysicsPrimitiveLayer',
        }

    def update_fps(self, fps: float):
        """Update FPS dan recompute window W. Panggil jika video FPS berubah."""
        self.fps = fps
        self._update_window()

    def reset(self):
        """Reset state antar video. Wajib dipanggil sebelum video baru."""
        self._iou_history.clear()
        self._pending_ce.clear()
        self._iou_contact_streak.clear()
        self._ce_raw_last_frame.clear()
        self._kelr_last_frame.clear()

    # ══════════════════════════════════════════════════════════════════════════
    # PRIMITIVE 1: Contact Event (CE)
    # Definisi: kontak kinematik pertama kali antara dua bounding box yang
    # sebelumnya terpisah, sambil masih dalam kondisi saling mendekat.
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_CE(self, track_i, track_j, frame_id: int) -> bool:
        """
        CE = (contact_detected) AND (approach_rate > 0) AND (rel_speed_ratio > 0.15)

        "contact_detected" didefinisikan sebagai SALAH SATU dari:
          1. IoU: 0 → positif  (kamera samping/isometric — overlap bbox terjadi)
          2. Center distance melewati threshold 1.5× max bbox diagonal  (kamera overhead —
             IoU bbox bisa tetap 0 karena kendaraan tampak dari atas tanpa overlap bbox)

        Threshold 1.5× bukan angka tuning — ini geometri: dua kendaraan bisa
        bersentuhan saat jarak pusat ≤ rata-rata half-diagonal masing-masing.
        1.5× memberikan sedikit margin untuk jitter deteksi YOLO.

        Guard rel_speed_ratio mencegah FP dari parallel driving / overtaking:
          - Parallel driving (kecepatan sama arah): rel_speed_ratio ≈ 0 → tidak fire
          - Head-on / rear-end: rel_speed_ratio → mendekati 1 → fire
        """
        pair_key = (min(track_i.track_id, track_j.track_id),
                    max(track_i.track_id, track_j.track_id))

        bbox_i = track_i.current_detection.get('bbox')
        bbox_j = track_j.current_detection.get('bbox')
        if bbox_i is None or bbox_j is None:
            return False

        iou_now = self._iou(bbox_i, bbox_j)

        # Ambil IoU frame sebelumnya dari history
        history = self._iou_history.get(pair_key, [])
        iou_prev = history[-1] if history else 0.0

        # Simpan IoU saat ini ke history (buffer pendek, hanya butuh prev)
        if pair_key not in self._iou_history:
            self._iou_history[pair_key] = []
        self._iou_history[pair_key].append(iou_now)
        if len(self._iou_history[pair_key]) > 3:
            self._iou_history[pair_key].pop(0)

        # ── Kondisi 1: Contact detection (camera-agnostic) ─────────────────────
        # Path A: IoU transition 0 → positif (kamera samping/isometric)
        iou_contact = (iou_now > 0.0 and iou_prev == 0.0)

        # Path B: Center distance < 1.5× max bbox diagonal (kamera overhead)
        # Fisika: dua kendaraan bersentuhan ketika jarak pusat ≤ half-diagonal masing-masing.
        # 1.5× memberikan margin untuk jitter YOLO. Scale-free: proporsional ke ukuran kendaraan.
        CE_CONTACT_DIAG_FACTOR = 1.5
        diag_i = math.sqrt(max(bbox_i[2] - bbox_i[0], 1) ** 2 + max(bbox_i[3] - bbox_i[1], 1) ** 2)
        diag_j = math.sqrt(max(bbox_j[2] - bbox_j[0], 1) ** 2 + max(bbox_j[3] - bbox_j[1], 1) ** 2)
        max_diag = max(diag_i, diag_j, 1.0)
        ci_now = track_i.current_detection.get('center')
        cj_now = track_j.current_detection.get('center')
        if ci_now is None:
            ci_now = [(bbox_i[0] + bbox_i[2]) / 2.0, (bbox_i[1] + bbox_i[3]) / 2.0]
        if cj_now is None:
            cj_now = [(bbox_j[0] + bbox_j[2]) / 2.0, (bbox_j[1] + bbox_j[3]) / 2.0]
        center_dist = math.sqrt((ci_now[0] - cj_now[0]) ** 2 + (ci_now[1] - cj_now[1]) ** 2)
        prox_contact = center_dist < max_diag * CE_CONTACT_DIAG_FACTOR

        if not (iou_contact or prox_contact):
            return False

        # ── Kondisi 2: approach_rate > 0 (sedang mendekat, bukan menjauh) ──────
        ci = self._get_raw_centers(track_i, last_n=2)
        cj = self._get_raw_centers(track_j, last_n=2)
        if len(ci) < 2 or len(cj) < 2:
            return False

        dist_now  = np.linalg.norm(np.array(ci[-1]) - np.array(cj[-1]))
        dist_prev = np.linalg.norm(np.array(ci[-2]) - np.array(cj[-2]))
        approach_rate = dist_prev - dist_now  # positif = mendekat
        if approach_rate <= 0:
            return False

        # ── Kondisi 3: rel_speed_ratio > 0.15 (bukan parallel / convoy) ────────
        v_i = self._raw_velocity(track_i)
        v_j = self._raw_velocity(track_j)
        total_speed = np.linalg.norm(v_i) + np.linalg.norm(v_j)
        if total_speed < 1e-6:
            # Kedua track tampak diam di raw velocity (Kalman freeze / kendaraan stop).
            # contact_detected dan approach_rate > 0 sudah dikonfirmasi di atas.
            # Keduanya bersama-sama adalah bukti geometric kontak yang cukup.
            # (Kendaraan parkir tidak punya approach_rate > 0 → aman.)
            return True
        rel_speed_ratio = np.linalg.norm(v_i - v_j) / total_speed

        return rel_speed_ratio > self.CE_MIN_SPEED_RATIO

    # ══════════════════════════════════════════════════════════════════════════
    # PRIMITIVE 2: Kinetic Energy Loss Ratio — Per-track Scalar (KELR_personal)
    # Diturunkan dari koefisien restitusi, Brach (1991).
    # Threshold 0.91 = 1 - e² dengan e = 0.3 (moderate crash).
    # PER-TRACK SCALAR: mengukur loss energi kinetik satu kendaraan (||v||, bukan v).
    # Scalar magnitude tidak dibatalkan oleh direction jitter (tidak seperti vector mean).
    #
    # Guard: EAGER-inspired Z-score (Andriyadi et al.)
    #   Speed drop harus > 2σ dari distribusi history track itu sendiri.
    #   std_speed mengabsorb jitter aktual → tidak ada magic number.
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_KELR_personal(self, track) -> bool:
        """
        KELR per-track (personal): deteksi sudden speed drop satu kendaraan.

        Berbeda dengan KELR_rel (pairwise, vector mean):
          - Menggunakan scalar speed (magnitude ||v||), bukan vector (v).
          - Scalar tidak dibatalkan oleh direction jitter.
          - Per-track: setiap kendaraan dinilai sendiri-sendiri.

        Dua tahap:
        1. KELR energy ratio (Brach 1991):
             speed_before = mean(||v||) dalam window W kecuali N_after terakhir
             speed_after  = mean(||v||) dari N_after frame terakhir
             KELR_personal = 1 - (speed_after² / speed_before²) > 0.91
        2. EAGER-inspired Z-score guard:
             z_drop = (mean_speed_before - speed_after) / std_speed_before > 2σ
             Self-normalizing: std mengabsorb jitter aktual track ini sendiri.

        Kenapa scalar lebih baik dari vector untuk per-track:
          Vector mean arah berubah-ubah (YOLO bbox jitter) → saling cancel → ≈ 0.
          Scalar mean selalu positif → tidak cancel → mendeteksi speed drop nyata. ✓
        Kenapa ByteTrack smooth tidak jadi masalah:
          Kecepatan dihitung dari raw centers di track.history, bukan Kalman state. ✓
        """
        centers = self._get_raw_centers(track, last_n=self.W + 1)
        if len(centers) < 4:
            return False, 0.0

        # 3-frame moving average on centers before computing speeds.
        # Category B FN fix: raw YOLO centers at the collision frame are a mix of
        # pre- and post-impact positions. This inflates speed_after, lowering KELR
        # below threshold even when the energy drop is real. Smoothing pulls the
        # transition frame toward its post-impact neighbours, giving a cleaner
        # speed_after ≈ 0 for stopped vehicles.
        if len(centers) >= 3:
            smoothed = []
            for k in range(len(centers)):
                lo = max(0, k - 1)
                hi = min(len(centers) - 1, k + 1)
                n_pts = hi - lo + 1
                cx = sum(centers[j][0] for j in range(lo, hi + 1)) / n_pts
                cy = sum(centers[j][1] for j in range(lo, hi + 1)) / n_pts
                smoothed.append((cx, cy))
            centers = smoothed

        # Scalar speeds per frame (magnitude ||v||), stride-2 untuk noise robustness.
        # Fisika: v = Δx / Δt, dengan Δt = 2 frame.
        # Keuntungan stride-2: jitter YOLO (A→B→A) cancel → speed ≈ 0 (benar).
        #                       random noise σ tereduksi √2.
        # Nilai dibagi 2 agar tetap dalam satuan px/frame (bukan px/2-frame).
        speeds = []
        for k in range(2, len(centers)):
            v = np.array(centers[k]) - np.array(centers[k - 2])
            speeds.append(float(np.linalg.norm(v)) / 2.0)

        # Adaptive n_after: gunakan setengah window agar post-collision frames
        # sepenuhnya masuk speeds_after, bukan mengotori mean_before.
        # Untuk track dengan history pendek, fallback ke KELR_N_AFTER minimum.
        n_after = max(self.KELR_N_AFTER, len(speeds) // 2)
        if len(speeds) < n_after + 2:
            return False, 0.0

        speeds_before = speeds[:-n_after]
        speeds_after  = speeds[-n_after:]

        mean_before = float(np.mean(speeds_before))
        speed_after = float(np.mean(speeds_after))

        if mean_before < 1e-3:
            # Track selalu diam sepanjang window → tidak ada yang bisa drop → bukan collision
            return False, 0.0

        # ── KELR energy ratio (Brach 1991) ──────────────────────────────────
        kelr_personal = 1.0 - (speed_after ** 2) / (mean_before ** 2)
        if kelr_personal <= self.KELR_THRESHOLD:
            return False, float(kelr_personal)

        # ── EAGER-inspired Z-score guard ────────────────────────────────────
        std_before = float(np.std(speeds_before))

        if std_before < 1e-6:
            # Zero variance: speed sangat stabil sebelumnya.
            # Sudden collapse setelah stable motion → strong signal.
            return mean_before > 1e-3, float(kelr_personal)

        z_drop = (mean_before - speed_after) / std_before
        return z_drop > self.KELR_Z_THRESHOLD, float(kelr_personal)

    # ══════════════════════════════════════════════════════════════════════════
    # PRIMITIVE 3: Aspect Ratio Shock (ARS)
    # Statistical hypothesis testing: reject null hypothesis (AR normal)
    # pada confidence level 95% (2σ).
    # was_moving guard (scale-free): mencegah FP dari YOLO bbox jitter
    # pada kendaraan diam / parkir.
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_ARS(self, track) -> bool:
        """
        AR_baseline = mean dan std dari AR selama N frame terakhir (sebelum sekarang)
        ARS = |AR_now - AR_baseline_mean| / AR_baseline_std  > 2σ
        Guard: was_moving = total_displacement_in_W > bbox_diagonal / 4

        was_moving guard adalah scale-free:
          - Truk besar: bbox_diagonal besar → threshold displacement juga besar
          - Motor kecil: bbox_diagonal kecil → threshold juga kecil proporsional
          → Tidak ada angka pixel absolut
        """
        if (not hasattr(track, 'history') or
                len(track.history) < self.AR_BASELINE_N + 1):
            return False, 0.0

        # Kumpulkan AR dari history
        ar_values = []
        for entry in track.history[-(self.AR_BASELINE_N + 1):]:
            bbox = entry.get('bbox')
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            w = max(float(x2 - x1), 1.0)
            h = max(float(y2 - y1), 1.0)
            ar_values.append(h / w)

        if len(ar_values) < 5:
            return False, 0.0

        ar_baseline = ar_values[:-1]   # semua kecuali frame sekarang
        ar_now      = ar_values[-1]

        mean_ar = float(np.mean(ar_baseline))
        std_ar  = float(np.std(ar_baseline))

        if std_ar < 1e-6:
            # AR track tidak pernah berubah = terlalu statis untuk diuji
            return False, 0.0

        z_score = abs(ar_now - mean_ar) / std_ar
        if z_score <= self.ARS_SIGMA:
            return False, float(z_score)

        # ── was_moving guard (scale-free) ────────────────────────────────────
        centers = self._get_raw_centers(track, last_n=self.W)
        if len(centers) < 2:
            return False, float(z_score)

        # Net displacement (start → end), bukan total path length.
        # total_displacement akumulasi jitter deteksi (±N px per frame × W frames)
        # sehingga kendaraan yang DIAM pun bisa lolos guard karena noise.
        # Net_displacement = ||center[-1] - center[0]|| → jitter saling cancel → filter noise.
        net_displacement = float(np.linalg.norm(
            np.array(centers[-1]) - np.array(centers[0])
        ))

        bbox = track.current_detection.get('bbox')
        if bbox is None:
            return False, float(z_score)
        x1, y1, x2, y2 = bbox
        bbox_diagonal = math.sqrt(max(x2 - x1, 1) ** 2 + max(y2 - y1, 1) ** 2)

        was_moving = net_displacement > (bbox_diagonal / 4.0)
        return was_moving, float(z_score)

    # ══════════════════════════════════════════════════════════════════════════
    # PRIMITIVE 4: Pairwise Momentum Violation (PMV)
    # Tabrakan = transfer momentum impulsif antara dua objek yang berdekatan.
    # Normal braking: p_sys turun bertahap (reaksi terhadap kondisi jalan).
    # Tabrakan: p_sys runtuh mendadak dalam 1-3 frame (pertukaran energi impulsif).
    # Z-score guard (sama dengan KELR) membedakan "mendadak" vs "bertahap".
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_PMV(self, track_i, track_j) -> Tuple[bool, float]:
        """
        Pairwise Momentum Violation: deteksi transfer momentum antara dua track.

        Fisika (Newton's 3rd law):
            Tabrakan   → Δp_A ≈ -Δp_B  (satu melambat, satu mendapat dorongan)
                       → Δp_i × Δp_j < 0  (berlawanan tanda)
            Normal braking → Δp_i < 0 DAN Δp_j < 0  (keduanya negatif, independen)
                           → Δp_i × Δp_j > 0  (tidak lolos)

        Proxy massa = bbox_area (scale-free).

        Returns (fired, transfer_score):
            transfer_score = min(|Δp_i|, |Δp_j|) / max(p_before_i, p_before_j)
            Menggambarkan seberapa besar momentum yang berpindah secara proporsional.
        """
        centers_i = self._get_raw_centers(track_i, last_n=self.W + 1)
        centers_j = self._get_raw_centers(track_j, last_n=self.W + 1)

        if len(centers_i) < 4 or len(centers_j) < 4:
            return False, 0.0

        # Scalar speeds stride-2 (noise-robust, sama dengan KELR_personal)
        speeds_i = []
        for k in range(2, len(centers_i)):
            speeds_i.append(float(np.linalg.norm(
                np.array(centers_i[k]) - np.array(centers_i[k-2]))) / 2.0)

        speeds_j = []
        for k in range(2, len(centers_j)):
            speeds_j.append(float(np.linalg.norm(
                np.array(centers_j[k]) - np.array(centers_j[k-2]))) / 2.0)

        n = min(len(speeds_i), len(speeds_j))
        if n < self.KELR_N_AFTER + 2:
            return False, 0.0

        speeds_i = speeds_i[-n:]
        speeds_j = speeds_j[-n:]

        # Proxy mass = bbox area (scale-free)
        bbox_i = track_i.current_detection.get('bbox', [0, 0, 1, 1])
        bbox_j = track_j.current_detection.get('bbox', [0, 0, 1, 1])
        area_i = max((bbox_i[2] - bbox_i[0]) * (bbox_i[3] - bbox_i[1]), 1.0)
        area_j = max((bbox_j[2] - bbox_j[0]) * (bbox_j[3] - bbox_j[1]), 1.0)

        # Before/after split
        mean_si_before = float(np.mean(speeds_i[:-self.KELR_N_AFTER]))
        mean_sj_before = float(np.mean(speeds_j[:-self.KELR_N_AFTER]))
        mean_si_after  = float(np.mean(speeds_i[-self.KELR_N_AFTER:]))
        mean_sj_after  = float(np.mean(speeds_j[-self.KELR_N_AFTER:]))

        # Momentum changes (mass-weighted, negatif = melambat, positif = mendapat dorongan)
        dp_i = area_i * (mean_si_after - mean_si_before)
        dp_j = area_j * (mean_sj_after - mean_sj_before)

        # Syarat utama: perubahan momentum BERLAWANAN tanda (Newton's 3rd law)
        if dp_i * dp_j >= 0:
            return False, 0.0

        # Setidaknya satu track harus punya perubahan signifikan (bukan noise)
        p_before_i = area_i * max(mean_si_before, 1e-3)
        p_before_j = area_j * max(mean_sj_before, 1e-3)
        rel_change_i = abs(dp_i) / p_before_i
        rel_change_j = abs(dp_j) / p_before_j

        # Kedua track harus berubah signifikan — satu saja tidak cukup (bisa jitter)
        if min(rel_change_i, rel_change_j) < self.PMV_MIN_REL_CHANGE:
            return False, 0.0

        # transfer_score: seberapa besar momentum yang berpindah secara proporsional
        transfer_score = min(abs(dp_i), abs(dp_j)) / max(p_before_i, p_before_j, 1.0)
        return True, float(transfer_score)

    # ══════════════════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _update_window(self):
        """W = fps × 1 detik, minimum 5 frame."""
        self.W = max(5, int(self.fps * 1.0))

    def _get_raw_centers(self, track, last_n: int) -> List:
        """
        Ambil raw centers dari track.history — BUKAN smoothed velocity tracker.
        Returns list of (cx, cy) tuples, ordered oldest → newest.
        """
        if not hasattr(track, 'history') or len(track.history) == 0:
            return []
        entries = track.history[-last_n:] if last_n > 0 else track.history
        return [e['center'] for e in entries if 'center' in e]

    def _net_displacement_in_window(self, track) -> float:
        """
        Net displacement (jarak start→end) dalam window W.

        Dipakai sebagai mobility indicator yang scale-free:
          - Kendaraan bergerak: net_disp >> 0
          - Kendaraan diam (termasuk jitter): net_disp ≈ 0

        Berbeda dari total_path_length: jitter px-per-frame saling cancel di net_disp,
        sehingga kendaraan yang benar-benar diam tidak dianggap "mobile".
        """
        centers = self._get_raw_centers(track, last_n=self.W)
        if len(centers) < 2:
            return 0.0
        return float(np.linalg.norm(np.array(centers[-1]) - np.array(centers[0])))

    def _raw_velocity(self, track) -> np.ndarray:
        """Instantaneous raw velocity dari 2 center terakhir (px/frame)."""
        centers = self._get_raw_centers(track, last_n=2)
        if len(centers) < 2:
            return np.array([0.0, 0.0])
        return np.array(centers[-1]) - np.array(centers[-2])

    def _net_displacement_vector(self, track) -> np.ndarray:
        """
        Net displacement VECTOR dari start ke end dalam window W.

        Berbeda dari _net_displacement_in_window (hanya magnitude) dan
        _raw_velocity (instantaneous, rentan jitter 1-frame).

        Dipakai A8.2 untuk angle check yang robust:
          - W frames ≈ 1 detik → jitter YOLO saling cancel → arah reliable
          - Kendaraan bergerak 10 px/fr: net_disp_vec ≈ [10*W, ...] → jelas
          - Kendaraan diam: net_disp_vec ≈ [0, 0] (A8.1 sudah exclude case ini)
        """
        centers = self._get_raw_centers(track, last_n=self.W)
        if len(centers) < 2:
            return np.array([0.0, 0.0])
        return np.array(centers[-1]) - np.array(centers[0])

    def _iou(self, bbox_a, bbox_b) -> float:
        """IoU antara dua bbox dalam format (x1, y1, x2, y2)."""
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = max((ax2 - ax1) * (ay2 - ay1), 1)
        area_b = max((bx2 - bx1) * (by2 - by1), 1)
        union  = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _has_enough_history(self, track) -> bool:
        """
        Track valid untuk Physics Primitive Layer jika:
        - Punya history minimal min(W+1, 5) frame
        - Sudah matched minimal 3 frame (hits >= 3) — mencegah noise dari track baru
        """
        return (
            hasattr(track, 'history') and
            len(track.history) >= min(self.W + 1, 5) and
            hasattr(track, 'hits') and
            track.hits >= 3
        )

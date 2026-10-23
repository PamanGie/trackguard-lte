# TrackGuard-LTE Production

Paket inference single-PC dan multi-node Jetson. Jalankan perintah dari root
repository dengan environment Python yang sesuai perangkat.

## Bobot YOLO dan dataset VisDrone

Bobot model dan dataset tidak disertakan dalam repository ini. Silakan latih
model deteksi YOLO sendiri menggunakan dataset VisDrone:

- Dataset resmi: https://github.com/VisDrone/VisDrone-Dataset
- Panduan format dan training: https://docs.ultralytics.com/datasets/detect/visdrone/
- Implementasi YOLO: https://github.com/ultralytics/ultralytics

Contoh training berikut adalah titik awal, bukan konfigurasi reproduksi hasil
paper. Sesuaikan arsitektur, hyperparameter, dan perangkat dengan kebutuhan:

```bash
yolo detect train model=yolo11n.pt data=VisDrone.yaml epochs=100 imgsz=640
```

Gunakan file `best.pt` hasil training untuk parameter `--model`, atau salin ke
root repository dengan nama `best_visdrone_full.pt` agar cocok dengan contoh
perintah di bawah. Model harus memakai nama kelas VisDrone yang sesuai;
perubahan bobot dapat mengubah hasil deteksi dan alarm. Ikuti ketentuan dataset
dan lisensi implementasi yang digunakan.

## Environment

Gunakan Python 3.10 dan environment inference yang sudah bekerja. PyTorch dan
torchvision harus sesuai CUDA/JetPack perangkat. Jangan mengganti wheel Jetson
dengan wheel desktop. `requirements-runtime.txt` mencatat dependensi langsung,
bukan lockfile environment eksperimen. FFmpeg diperlukan untuk sumber video
yang memakainya; broker MQTT (misalnya Mosquitto) dijalankan terpisah di central.

## Single-PC: collision inference

```bash
python main.py --source "PATH_VIDEO_ATAU_URL" --model best_visdrone_full.pt --tracker bytetrack --conf 70 --physics --detect collision --no-display --no-output --evidence-dir runtime/single/evidence
```

Hapus `--no-display` untuk tampilan video. Snapshot berada di perangkat inference,
di `runtime/single/evidence/red`, `critical`, dan `caution`.

## Multi-node

1. Jalankan broker di central pada port 1883, dapat diakses ketiga Jetson.
2. Jalankan receiver sebelum node. Ganti `CENTRAL_PC_IP` dengan alamat broker.

```bash
python multi-nodes/central/main_central.py --mqtt-host CENTRAL_PC_IP --mqtt-port 1883 --sync-interval 5 --log-interval 1 --timeout 0 --output-dir runtime/central
```

3. Jalankan satu node pada setiap Jetson dari folder paket ini:

```bash
python multi-nodes/jetson/main_trackguard_node.py --source "https://tcnvr3.taichung.gov.tw/b07122e5" --model best_visdrone_full.pt --tracker bytetrack --conf 70 --jetson-id A --camera-id taichung_A --mqtt-host CENTRAL_PC_IP --mqtt-port 1883 --device-mode cuda --stage-profile-dir runtime/node_A --evidence-dir runtime/evidence_A --resource-sample-interval 2 --reporting-mode mqtt --episode-gap-seconds 10 --live-refresh-interval 60
```

```bash
python multi-nodes/jetson/main_trackguard_node.py --source "https://tcnvr6.taichung.gov.tw/0a27f44a" --model best_visdrone_full.pt --tracker bytetrack --conf 70 --jetson-id B --camera-id taichung_B --mqtt-host CENTRAL_PC_IP --mqtt-port 1883 --device-mode cuda --stage-profile-dir runtime/node_B --evidence-dir runtime/evidence_B --resource-sample-interval 2 --reporting-mode mqtt --episode-gap-seconds 10 --live-refresh-interval 60
```

```bash
python multi-nodes/jetson/main_trackguard_node.py --source "https://tcnvr6.taichung.gov.tw/8582fc8e" --model best_visdrone_full.pt --tracker bytetrack --conf 70 --jetson-id C --camera-id taichung_C --mqtt-host CENTRAL_PC_IP --mqtt-port 1883 --device-mode cuda --stage-profile-dir runtime/node_C --evidence-dir runtime/evidence_C --resource-sample-interval 2 --reporting-mode mqtt --episode-gap-seconds 10 --live-refresh-interval 60
```

URL di atas adalah konfigurasi eksperimen sebelumnya; ketersediaan stream saat
deployment harus diuji. Untuk uji 5 menit di Linux, tambahkan
`timeout --signal=INT 300s` sebelum `python`; 24 jam memakai `86400s`.
Mode `--mqtt-smoke-test` hanya menguji pesan MQTT, tidak melakukan inference
atau menghasilkan snapshot. Verifikasi penerimaan pesan di central.

Snapshot `runtime/evidence_A`, `evidence_B`, `evidence_C` tersimpan di Jetson
masing-masing dengan subfolder `red/critical/caution`. MQTT mengirim metadata,
bukan file PNG. CSV receiver tersimpan di central.

## Cakupan pemeriksaan

Paket ini memisahkan runtime dan bobot dari dataset, hasil eksperimen, arsip,
skrip training/evaluasi, serta entry point lama `main_jetson.py` yang merujuk
modul yang tidak tersedia. Source dan data asal tetap dipertahankan.
Verifikasi lokal: 104 file Python lolos parsing sintaks; CLI node dan central
berhasil dibuka; import ByteTrack dan pemuatan bobot berhasil; prediksi CPU pada
satu frame sintetis berhasil (10 kelas). Pengujian ini belum mencakup inference
CUDA, stream CCTV, atau MQTT end-to-end; uji perangkat sebelum deployment.

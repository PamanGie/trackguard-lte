"""
Main script for central PC.
Subscribes to MQTT messages from Jetson devices and logs results.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Import using direct file imports (to handle dash in folder name)
import importlib.util

# Import MQTT subscriber
subscriber_path = Path(__file__).parent / "mqtt_subscriber.py"
spec = importlib.util.spec_from_file_location("mqtt_subscriber", subscriber_path)
subscriber_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subscriber_module)
MQTTSubscriber = subscriber_module.MQTTSubscriber

# Import network monitor
network_path = Path(__file__).parent / "network_monitor.py"
spec = importlib.util.spec_from_file_location("network_monitor", network_path)
network_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(network_module)
NetworkMonitor = network_module.NetworkMonitor

# Import data aggregator
aggregator_path = Path(__file__).parent / "data_aggregator.py"
spec = importlib.util.spec_from_file_location("data_aggregator", aggregator_path)
aggregator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregator_module)
DataAggregator = aggregator_module.DataAggregator

# Import results logger
logger_path = Path(__file__).parent / "results_logger.py"
spec = importlib.util.spec_from_file_location("results_logger", logger_path)
logger_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logger_module)
ResultsLogger = logger_module.ResultsLogger

# Import timestamp utils
timestamp_path = Path(__file__).parent.parent / "utils" / "timestamp_utils.py"
spec = importlib.util.spec_from_file_location("timestamp_utils", timestamp_path)
timestamp_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timestamp_module)
TimestampUtils = timestamp_module.TimestampUtils


def parse_clock_offsets(value: Optional[str]) -> Dict[str, float]:
    """Parse clock offsets like 'A=84.8,B=84.8,C=84.8' or 'all=84.8'."""
    offsets: Dict[str, float] = {}
    if not value:
        return offsets
    for part in value.split(','):
        part = part.strip()
        if not part:
            continue
        if '=' in part:
            key, raw = part.split('=', 1)
            offsets[key.strip()] = float(raw.strip())
        else:
            offsets['all'] = float(part)
    return offsets
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LTE-TrackGuard Central PC Multi-Node Evaluation")
    parser.add_argument(
        "--mqtt-host",
        type=str,
        default="localhost",
        help="MQTT broker host"
    )
    parser.add_argument(
        "--mqtt-port",
        type=int,
        default=1883,
        help="MQTT broker port"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for CSV files (auto-detect from message if not specified)"
    )
    parser.add_argument(
        "--experiment-mode",
        type=str,
        choices=["auto", "event-based", "full-stream", "cpu"],
        default="auto",
        help="Experiment mode for output directory (auto = detect from message, or specify manually)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Log statistics every N messages"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Auto-exit if no messages received for N seconds (default: 30)"
    )
    parser.add_argument(
        "--clock-offset-ms",
        type=str,
        default=None,
        help="Clock correction in ms, e.g. all=84.8 or A=84.8,B=84.8,C=84.8"
    )
    parser.add_argument(
        "--sync-interval",
        type=float,
        default=300.0,
        help="Seconds between MQTT clock-sync pings (default: 300)"
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=None,
        help="Optional auto-stop duration in hours, e.g. 3"
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Determine output directory based on experiment mode
    output_dir = args.output_dir
    if output_dir is None:
        if args.experiment_mode == "auto":
            # Will be auto-detected from first message
            output_dir = "results"  # Temporary default
        elif args.experiment_mode == "event-based":
            output_dir = "results"
        elif args.experiment_mode == "full-stream":
            output_dir = "full_stream"
        elif args.experiment_mode == "cpu":
            output_dir = "jetson"
    
    print("=" * 70)
    print("LTE-TRACKGUARD MULTI-NODE EVALUATION - CENTRAL PC")
    print("=" * 70)
    print(f"MQTT Broker: {args.mqtt_host}:{args.mqtt_port}")
    print(f"Experiment Mode: {args.experiment_mode}")
    print(f"Output Directory: {output_dir}")
    clock_offsets_ms = parse_clock_offsets(args.clock_offset_ms)
    if clock_offsets_ms:
        print(f"Clock correction: {clock_offsets_ms} ms")
    print("=" * 70)
    
    # Initialize components
    print("\n[1/4] Connecting to MQTT broker...")
    subscriber = MQTTSubscriber(broker_host=args.mqtt_host, broker_port=args.mqtt_port)
    if not subscriber.connect():
        print("ERROR: Failed to connect to MQTT broker. Exiting.")
        return
    print("OK: Connected to MQTT broker")
    
    print("\n[2/4] Initializing network monitor...")
    network_monitor = NetworkMonitor(clock_offsets_ms=clock_offsets_ms)
    print("OK: Network monitor initialized")
    
    print("\n[3/4] Initializing data aggregator...")
    data_aggregator = DataAggregator()
    print("OK: Data aggregator initialized")
    
    print("\n[4/4] Initializing results logger...")
    results_logger = ResultsLogger(output_dir=output_dir)
    print("OK: Results logger initialized")
    print(f"  Results will be saved to: {output_dir}")
    print("  WARNING: Do not open CSV files in Excel while program is running!")
    
    # Track if we need to re-init logger after auto-detect
    detected_mode = None
    logger_reinitialized = False
    
    print("\n" + "=" * 70)
    print("LISTENING FOR MESSAGES")
    print("=" * 70)
    print("Press Ctrl+C to stop\n")
    
    start_time = time.time()
    message_count = 0
    sync_seq = 0
    last_sync_time = 0.0
    sync_offsets = {}
    
    def send_sync_ping():
        nonlocal sync_seq, last_sync_time
        sync_seq += 1
        t0 = time.time()
        msg = {
            "message_type": "sync_ping",
            "sync_seq": sync_seq,
            "server_send_time": t0,
        }
        ok = subscriber.publish_json("trackguard/sync/all", msg, qos=1)
        last_sync_time = t0
        if not ok:
            print(f"[sync] failed to publish sync ping #{sync_seq}")

    def on_message_received(message: dict):
        """Callback when message is received."""
        nonlocal message_count, last_message_time, detected_mode, logger_reinitialized, results_logger, output_dir
        
        message_type = message.get("message_type")
        if message_type == "sync_response":
            try:
                t0 = float(message.get("server_send_time"))
                t1 = float(message.get("jetson_receive_time"))
                t2 = float(message.get("jetson_send_time"))
                t3 = float(message.get("receive_time"))
                jetson_id = str(message.get("jetson_id", "unknown"))
                offset_s = ((t1 - t0) + (t2 - t3)) / 2.0
                rtt_s = (t3 - t0) - (t2 - t1)
                offset_ms = offset_s * 1000.0
                rtt_ms = rtt_s * 1000.0
                network_monitor.update_clock_offset(jetson_id, offset_ms)
                sync_offsets[jetson_id] = offset_ms
                print(f"[sync] Jetson {jetson_id}: offset={offset_ms:.3f} ms, rtt={rtt_ms:.3f} ms")
            except Exception as e:
                print(f"[sync] failed to process sync response: {e}")
            return

        # Auto-detect mode from first message if needed
        if args.output_dir is None and args.experiment_mode == "auto" and not logger_reinitialized:
            streaming_mode = message.get("streaming_mode", "event-based")
            device_mode = message.get("device_mode", "cuda")
            
            # Determine output directory based on detected mode
            if device_mode == "cpu":
                detected_mode = "cpu"
                new_output_dir = "jetson"
            elif streaming_mode == "full-stream":
                detected_mode = "full-stream"
                new_output_dir = "full_stream"
            else:
                detected_mode = "event-based"
                new_output_dir = "results"
            
            # Re-initialize logger with correct output directory
            if new_output_dir != output_dir:
                print(f"\n[Auto-detect] Detected mode: {detected_mode}")
                print(f"  Switching output directory: {output_dir} -> {new_output_dir}")
                results_logger.close()
                output_dir = new_output_dir
                results_logger = ResultsLogger(output_dir=output_dir)
                print(f"  Results will now be saved to: {output_dir}")
            
            logger_reinitialized = True
        
        message_count += 1
        last_message_time = time.time()  # Update last message time

        if message_count == 1 or message_type in {"alarm", "status"}:
            print(
                f"[mqtt] #{message_count} topic={message.get('_mqtt_topic', '')} "
                f"type={message_type or 'legacy'} jetson={message.get('jetson_id', 'unknown')} "
                f"camera={message.get('camera_id', '')} frame={message.get('frame_id', 0)}"
            )
        
        # Estimate message size (rough approximation)
        message_size = len(json.dumps(message).encode('utf-8'))
        
        # Record in network monitor
        network_monitor.record_message(message, message_size)
        
        # Add to data aggregator
        data_aggregator.add_message(message)
        
        # Calculate network stats for this message
        jetson_id = message.get("jetson_id", "unknown")
        net_stats = network_monitor.get_network_latency_stats(jetson_id)
        e2e_stats = network_monitor.get_end_to_end_latency_stats(jetson_id)
        
        # Log frame result (with error handling)
        try:
            frame_stats = {
                "network_latency_ms": net_stats.get("avg_ms", 0.0),
                "end_to_end_latency_ms": e2e_stats.get("avg_ms", 0.0),
                "payload_bytes": message_size
            }
            results_logger.log_frame_result(message, frame_stats)
        except (PermissionError, IOError) as e:
            # File might be open in Excel - log warning but continue
            if message_count % 100 == 0:  # Only warn every 100 messages
                print(f"WARNING: Could not write to CSV (file may be open): {e}")
        
        # Periodic statistics
        if message_count % args.log_interval == 0:
            elapsed = time.time() - start_time
            throughput = message_count / elapsed if elapsed > 0 else 0
            
            print(f"\n[{message_count} messages received]")
            print(f"  Throughput: {throughput:.2f} msg/sec")
            print(f"  Elapsed time: {elapsed:.1f} seconds")
            
            # Print per-Jetson stats
            all_net_stats = network_monitor.get_all_stats()
            all_resource_stats = data_aggregator.get_all_stats()
            
            for jetson_id in sorted(all_net_stats.keys()):
                net = all_net_stats[jetson_id]
                res = all_resource_stats.get(jetson_id, {})
                res_metrics = res.get("resource", {})
                
                print(f"\n  Jetson {jetson_id}:")
                print(f"    Network latency: {net['network_latency']['avg_ms']:.2f} ms")
                print(f"    E2E latency: {net['end_to_end_latency']['avg_ms']:.2f} ms")
                print(f"    Bandwidth: {net['bandwidth']['kbps']:.2f} kbps")
                alarms = res.get("alarms", {})
                if alarms:
                    print(f"    Alarms: total={alarms.get('total_alarm_events', 0)} "
                          f"red={alarms.get('red_events', 0)} "
                          f"warning={alarms.get('warning_events', 0)} "
                          f"caution={alarms.get('caution_events', 0)}")
                # Note: Resource metrics (GPU/RAM/Power) are measured locally on Jetson
                # and displayed in Jetson's terminal summary, not sent via MQTT
                
                # Display behavior analytics from latest message
                confirmed = message.get("confirmed_particles", [])
                if confirmed:
                    # Aggregate behavior analytics
                    size_statuses = {}
                    spread_directions = {}
                    for p_data in confirmed:
                        behavior = p_data.get("behavior_analytics", {})
                        size_status = behavior.get("size_status", "STABLE")
                        spread_dir = behavior.get("spread_direction", "STATIC")
                        
                        size_statuses[size_status] = size_statuses.get(size_status, 0) + 1
                        spread_directions[spread_dir] = spread_directions.get(spread_dir, 0) + 1
                    
                    # Format behavior summary
                    if size_statuses or spread_directions:
                        size_str = ", ".join([f"{k}:{v}" for k, v in size_statuses.items()])
                        dir_str = ", ".join([f"{k}:{v}" for k, v in spread_directions.items()])
                        print(f"    Behavior: Size[{size_str}] | Direction[{dir_str}]")
    
    # Set callback
    subscriber.set_callback(on_message_received)
    
    # Track last message time for timeout detection
    last_message_time = time.time()
    if args.sync_interval > 0:
        print(f"Clock sync active: MQTT ping every {args.sync_interval:.1f} seconds")
        send_sync_ping()
    
    try:
        # Keep running with timeout detection
        while True:
            time.sleep(1)
            now_loop = time.time()
            if args.duration_hours is not None and (now_loop - start_time) >= args.duration_hours * 3600.0:
                print(f"\nReached requested duration: {args.duration_hours:.3f} h. Stopping...")
                break
            if args.sync_interval > 0 and (now_loop - last_sync_time) >= args.sync_interval:
                send_sync_ping()
            
            # Messages are processed by the MQTT callback. Do not drain the
            # backup queue here, otherwise each message is counted twice.
            if True:
                # Check timeout - if no messages for X seconds, assume Jetson finished
                # Only check timeout if we've received at least one message
                if message_count > 0:
                    time_since_last = time.time() - last_message_time
                    if time_since_last > args.timeout:
                        print(f"\nWARNING: No messages received for {time_since_last:.1f} seconds.")
                        print(f"   Assuming Jetson(s) finished processing. Exiting...")
                        break
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Final statistics and logging
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("FINAL STATISTICS")
        print("=" * 70)
        
        # Get all statistics
        all_net_stats = network_monitor.get_all_stats()
        all_resource_stats = data_aggregator.get_all_stats()
        total_stats = data_aggregator.get_total_stats()
        
        print(f"\nTotal messages received: {message_count}")
        print(f"Total time: {elapsed:.2f} seconds")
        print(f"Average throughput: {message_count / elapsed:.2f} msg/sec" if elapsed > 0 else "Average throughput: N/A")
        current_offsets = network_monitor.get_clock_offsets()
        if current_offsets:
            print(f"Clock offsets used (Jetson minus server, ms): {current_offsets}")
        
        # Print per-Jetson final stats
        print("\nPer-Jetson Statistics:")
        for jetson_id in sorted(all_net_stats.keys()):
            net = all_net_stats[jetson_id]
            res = all_resource_stats.get(jetson_id, {})
            res_metrics = res.get("resource", {})
            proc_metrics = res.get("processing", {})
            
            print(f"\n  Jetson {jetson_id}:")
            print(f"    Network Latency: {net['network_latency']['avg_ms']:.2f} ms (min: {net['network_latency']['min_ms']:.2f}, max: {net['network_latency']['max_ms']:.2f})")
            print(f"    E2E Latency: {net['end_to_end_latency']['avg_ms']:.2f} ms (min: {net['end_to_end_latency']['min_ms']:.2f}, max: {net['end_to_end_latency']['max_ms']:.2f})")
            print(f"    Bandwidth: {net['bandwidth']['kbps']:.2f} kbps ({net['bandwidth']['total_bytes']} bytes total)")
            print(f"    Throughput: {net['throughput_msg_per_sec']:.2f} msg/sec")
            print(f"    Frames: {proc_metrics.get('total_frames', 0)}")
            alarms = res.get("alarms", {})
            print(f"    Alarms: total={alarms.get('total_alarm_events', 0)} "
                  f"red={alarms.get('red_events', 0)} "
                  f"warning={alarms.get('warning_events', 0)} "
                  f"caution={alarms.get('caution_events', 0)}")
            # Note: Resource metrics (GPU/RAM/Power) are measured locally on Jetson
            # and displayed in Jetson's terminal summary, not sent via MQTT
        
        # Print total stats
        print("\nTotal System Statistics:")
        print(f"  Total Frames: {total_stats['total_frames']}")
        print(f"  Number of Jetsons: {total_stats['num_jetsons']}")
        # Note: Resource metrics (GPU/RAM/Power) are measured locally on each Jetson
        # and displayed in each Jetson's terminal summary, not aggregated via MQTT
        
        # Log final statistics (with error handling)
        print("\nLogging final statistics to CSV...")
        try:
            results_logger.log_network_stats(all_net_stats)
        except (PermissionError, IOError) as e:
            print(f"WARNING: Could not write network stats: {e}")
        
        try:
            results_logger.log_resource_stats(all_resource_stats)
        except (PermissionError, IOError) as e:
            print(f"WARNING: Could not write resource stats: {e}")
        
        # Close files (with error handling)
        results_logger.close()
        
        output_files = results_logger.get_output_files()
        print("\nResults saved to:")
        for file_type, file_path in output_files.items():
            print(f"  {file_type}: {file_path}")
        
        # Disconnect
        subscriber.disconnect()
        
        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    main()



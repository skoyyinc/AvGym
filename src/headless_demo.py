#!/usr/bin/env python3
"""
Headless Active Vision Demo for SO-ARM101
Text-based demo without GUI dependencies
"""

import time
import numpy as np
from active_vision_system import ActiveVisionSystem

class HeadlessActiveVisionSystem(ActiveVisionSystem):
    """Active vision system without GUI display"""
    
    def _visualize_results(self, image: np.ndarray, detections):
        """Text-based visualization instead of GUI"""
        # Print status every few frames to avoid spam
        if not hasattr(self, '_last_print') or time.time() - self._last_print > 1.0:
            status = (f"📊 [Camera {self.camera_controller.active_camera_id}] "
                     f"Objects: {len(detections)} | "
                     f"Targets: {len(self.current_targets)} | "
                     f"Switches: {self.camera_switches} | "
                     f"Occlusion: {'⚠️' if self.occlusion_detected else '✅'}")
            print(status)
            self._last_print = time.time()
    
    def start_active_vision(self):
        """Start headless active vision loop"""
        print("🚀 Starting headless active vision loop...")
        print("Press Ctrl+C to stop")
        self.tracking_active = True
        
        try:
            step_count = 0
            while self.tracking_active:
                self._active_vision_step()
                step_count += 1
                
                # Show progress every 100 steps
                if step_count % 100 == 0:
                    metrics = self.get_performance_metrics()
                    print(f"📈 Step {step_count}: Processed {metrics['detection_count']} detections")
                
                time.sleep(0.1)  # 10 FPS
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopping active vision...")
            self.tracking_active = False

def main():
    print("🤖 SO-ARM101 Headless Active Vision Demo")
    print("=" * 50)
    
    config = {
        'robot_type': 'so_arm101',
        'simulation_mode': True,
        'headless': True
    }
    
    av_system = HeadlessActiveVisionSystem(config)
    
    print("\n📋 Headless Demo Features:")
    print("- Text-based status updates")
    print("- Active vision without GUI")
    print("- Performance metrics tracking")
    print("- Occlusion detection simulation")
    
    try:
        av_system.start_active_vision()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n📊 Final Performance Metrics:")
        metrics = av_system.get_performance_metrics()
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        print("\n👋 Headless demo completed!")

if __name__ == "__main__":
    main()
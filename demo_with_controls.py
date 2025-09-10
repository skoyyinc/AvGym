#!/usr/bin/env python3
"""
Enhanced SO-ARM101 Object Tracking Demo with Control Instructions - FIXED
"""

import time
import numpy as np
import pybullet as p
from src.so_arm_gym_env import SO101CameraTrackingEnv
from src.tracking_policy import ImprovedTrackingPolicy

class EnhancedTrackingDemo:
    """Enhanced demo with better controls and visualization"""
    
    def __init__(self):
        self.env = None
        self.policy = None
        self.gui_ids = {}
        self.physics_connected = False
        
    def print_controls(self):
        """Print control instructions"""
        print("\n🎮 DEMO CONTROLS")
        print("=" * 50)
        print("🖱️  MOUSE CONTROLS:")
        print("   • Left-click + drag: Rotate camera view")
        print("   • Right-click + drag: Pan camera")
        print("   • Mouse wheel: Zoom in/out")
        print("   • Shift + drag: Alternative pan")
        
        print("\n⌨️  KEYBOARD SHORTCUTS:")
        print("   • R: Reset camera to default view")
        print("   • G: Toggle debug GUI panels")
        print("   • W: Toggle wireframe mode")
        print("   • P: Pause/unpause simulation")
        print("   • Space: Single step (when paused)")
        print("   • S: Take screenshot")
        print("   • ESC/Q: Quit demo")
        
        print("\n🎛️  GUI SLIDERS (will appear in PyBullet window):")
        print("   • Demo Speed: Control simulation speed")
        print("   • Tracking Gain: Adjust robot responsiveness")
        print("   • Pause: Pause/unpause demo")
        print("   • Reset Scene: Reset robot and target")
        
        print("\n📊 WHAT TO WATCH:")
        print("   • Red sphere: Target object to track")
        print("   • Robot arm: Will move to center target in camera")
        print("   • Green text: Target visible and being tracked")
        print("   • Red text: Target lost, robot searching")
        
        print("=" * 50)
        
    def setup_gui(self):
        """Setup enhanced GUI - call after physics server is connected"""
        if not self.physics_connected:
            print("⚠️  Cannot setup GUI - physics server not connected")
            return
            
        print("🎮 Setting up enhanced GUI...")
        
        try:
            # Control sliders
            self.gui_ids['speed_slider'] = p.addUserDebugParameter(
                "Demo Speed", 0.1, 2.0, 1.0
            )
            
            self.gui_ids['gain_slider'] = p.addUserDebugParameter(
                "Tracking Gain", 0.1, 2.0, 0.8
            )
            
            # Control buttons
            self.gui_ids['pause_button'] = p.addUserDebugParameter(
                "Pause (1=pause)", 0, 1, 0
            )
            
            self.gui_ids['reset_button'] = p.addUserDebugParameter(
                "Reset Scene", 0, 1, 0
            )
            
            print("✅ GUI controls added to PyBullet window")
            
        except Exception as e:
            print(f"⚠️  GUI setup failed: {e}")
            print("   Demo will continue without GUI controls")
        
    def run_demo(self):
        """Run enhanced demo"""
        print("🚀 Starting Enhanced SO-ARM101 Tracking Demo")
        self.print_controls()
        
        try:
            # Create environment with GUI - this connects to physics server
            print("\n📡 Connecting to physics server...")
            self.env = SO101CameraTrackingEnv(render_mode="human")
            self.physics_connected = True
            print("✅ Physics server connected")
            
            # Create policy
            print("🧠 Creating tracking policy...")
            self.policy = ImprovedTrackingPolicy({
                'p_gain_fast': 1.2,    # Increased from 0.8 for faster response
                'p_gain_slow': 0.5,    # Increased from 0.3 for quicker fine adjustments
                'max_velocity': 1.3    # Increased from 1.0 for higher max speed
            })
            
            # Reset environment first
            print("🔄 Resetting environment...")
            observation, info = self.env.reset()
            
            # Now setup GUI (after physics server is connected)
            self.setup_gui()
            
            # Add visual markers
            self._add_visual_markers()
            
            print("\n🎬 Demo is now running in PyBullet GUI window!")
            print("   Use mouse and keyboard controls to interact")
            print("   Close PyBullet window or press Ctrl+C to stop")
            
            # Main demo loop
            self._run_demo_loop(observation)
            
        except Exception as e:
            print(f"❌ Demo error: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            if self.env:
                self.env.close()
            print("\n👋 Demo ended!")
    
    def _run_demo_loop(self, observation):
        """Main demo loop"""
        step_count = 0
        last_reset_state = 0
        
        # Track GUI state
        gui_available = bool(self.gui_ids)
        
        while True:
            try:
                # Default values
                demo_speed = 1.0
                tracking_gain = 0.8
                pause_state = 0
                reset_state = 0
                
                # Read GUI parameters if available
                if gui_available:
                    try:
                        demo_speed = p.readUserDebugParameter(self.gui_ids['speed_slider'])
                        tracking_gain = p.readUserDebugParameter(self.gui_ids['gain_slider'])
                        pause_state = p.readUserDebugParameter(self.gui_ids['pause_button'])
                        reset_state = p.readUserDebugParameter(self.gui_ids['reset_button'])
                    except:
                        # GUI might have been closed
                        gui_available = False
                
                # Handle reset
                if reset_state > 0.5 and last_reset_state < 0.5:
                    print("🔄 Resetting scene...")
                    observation, info = self.env.reset()
                    self._add_visual_markers()
                    step_count = 0
                last_reset_state = reset_state
                
                # Handle pause
                if pause_state > 0.5:
                    time.sleep(0.1)
                    continue
                
                # Update policy gain
                self.policy.p_gain_fast = tracking_gain
                
                # Get action and step
                action = self.policy.predict(observation)
                observation, reward, terminated, truncated, info = self.env.step(action)
                
                # Update status display
                self._update_status_display(observation, step_count, reward)
                
                step_count += 1
                
                # Print status every 50 steps
                if step_count % 50 == 0:
                    target_status = "VISIBLE" if observation['target_in_view'][0] > 0.5 else "SEARCHING"
                    error = observation['target_center_distance'][0]
                    print(f"📊 Step {step_count}: Target {target_status}, Error: {error:.3f}, Reward: {reward:.1f}")
                
                # Control demo speed
                time.sleep(0.1 / demo_speed)
                
                if terminated or truncated:
                    print("📋 Episode finished, resetting...")
                    observation, info = self.env.reset()
                    self._add_visual_markers()
                    step_count = 0
                    
            except KeyboardInterrupt:
                break
    
    def _add_visual_markers(self):
        """Add visual markers to help understand the demo"""
        if not self.physics_connected:
            return
            
        try:
            # Add coordinate frame at origin
            p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], lineWidth=3)  # X-axis red
            p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], lineWidth=3)  # Y-axis green  
            p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], lineWidth=3)  # Z-axis blue
            
            # Add workspace boundary circle
            points = []
            for i in range(20):
                angle = 2 * np.pi * i / 20
                x = 0.5 * np.cos(angle)
                y = 0.5 * np.sin(angle)
                points.append([x, y, 0])
            
            for i in range(len(points)):
                next_i = (i + 1) % len(points)
                p.addUserDebugLine(points[i], points[next_i], [0.5, 0.5, 0.5], lineWidth=2)
                
        except Exception as e:
            print(f"⚠️  Could not add visual markers: {e}")
        
    def _update_status_display(self, observation, step, reward):
        """Update status text display"""
        if not self.physics_connected:
            return
            
        try:
            target_visible = observation['target_in_view'][0] > 0.5
            center_distance = observation['target_center_distance'][0]
            
            status_color = [0, 1, 0] if target_visible else [1, 0, 0]
            
            status_text = (
                f"Step: {step} | "
                f"Target: {'VISIBLE' if target_visible else 'SEARCHING'} | "
                f"Error: {center_distance:.3f} | "
                f"Reward: {reward:.1f}"
            )
            
            # Remove old status text
            if hasattr(self, '_status_text_id'):
                try:
                    p.removeUserDebugItem(self._status_text_id)
                except:
                    pass  # Text might have been removed already
            
            # Add new status text
            self._status_text_id = p.addUserDebugText(
                status_text,
                [0, 0, 0.8],
                textColorRGB=status_color,
                textSize=1.5
            )
            
        except Exception as e:
            # Silently fail if text display doesn't work
            pass

def main():
    """Main function"""
    demo = EnhancedTrackingDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()
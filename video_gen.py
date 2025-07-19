import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import os
import gc
from PIL import Image

# Try to import video generation dependencies
try:
    from diffusers import TextToVideoSDPipeline, DiffusionPipeline
except ImportError:
    TextToVideoSDPipeline = None
    DiffusionPipeline = None

# Available video models
AVAILABLE_VIDEO_MODELS = {
    "Zeroscope v2 576w": "cerspense/zeroscope_v2_576w",
    "ModelScope T2V": "damo-vilab/text-to-video-ms-1.7b",
    "Text-to-Video Zero": "text-to-video-zero"
}

@st.cache_resource
def load_video_model(model_id, device_strategy="auto", dtype=torch.float16):
    """Load and cache the video generation model"""
    try:
        st.info(f"Loading video model: {model_id}")
        
        # Try different loading approaches
        pipeline = None
        load_approaches = [
            ("with safetensors", {"use_safetensors": True}),
            ("without safetensors", {"use_safetensors": False}),
            ("default loading", {})
        ]
        
        for approach_name, load_kwargs in load_approaches:
            try:
                base_kwargs = {
                    "torch_dtype": dtype,
                    "low_cpu_mem_usage": True,
                }
                base_kwargs.update(load_kwargs)
                
                if "zeroscope" in model_id.lower():
                    pipeline = TextToVideoSDPipeline.from_pretrained(
                        model_id,
                        **base_kwargs
                    )
                else:
                    pipeline = DiffusionPipeline.from_pretrained(
                        model_id,
                        **base_kwargs
                    )
                
                st.info(f"✅ Loaded {approach_name}")
                break
                
            except Exception as approach_error:
                st.info(f"⚠️ {approach_name} failed: {str(approach_error)[:100]}...")
                continue
        
        if pipeline is None:
            raise Exception("All loading approaches failed")
        
        # Apply memory optimizations
        if device_strategy == "cpu_offload" and torch.cuda.is_available():
            pipeline.enable_model_cpu_offload()
            pipeline.enable_attention_slicing()
            if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                except:
                    pass
        elif device_strategy == "cuda" and torch.cuda.is_available():
            pipeline = pipeline.to("cuda")
            pipeline.enable_attention_slicing()
            if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                except:
                    pass
        else:
            pipeline = pipeline.to("cpu")
        
        # Test the pipeline
        if torch.cuda.is_available() and device_strategy != "cpu":
            torch.cuda.synchronize()
        
        st.success(f"✅ Model loaded successfully!")
        return pipeline
        
    except Exception as e:
        st.error(f"❌ Error loading model {model_id}: {str(e)}")
        st.error("This might be due to:")
        st.error("- Network connectivity issues")
        st.error("- Invalid model ID")
        st.error("- Insufficient memory")
        st.error("- Missing dependencies")
        st.error("- Model not compatible with current diffusers version")
        return None

def generate_video(pipeline, prompt, params):
    """Generate a video with given parameters"""
    try:
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Prepare generation parameters
        generation_kwargs = {
            "prompt": prompt,
            "num_frames": params["num_frames"],
            "num_inference_steps": params["steps"],
            "guidance_scale": params["guidance"],
            "height": params["height"],
            "width": params["width"],
        }
        
        # Handle seed
        if params["seed"] is not None:
            generator = torch.Generator()
            if torch.cuda.is_available() and hasattr(pipeline, 'device') and 'cuda' in str(pipeline.device):
                generator = torch.Generator(device='cuda')
            generator.manual_seed(params["seed"])
            generation_kwargs["generator"] = generator
        
        # Generate video
        progress_placeholder = st.empty()
        progress_placeholder.info("🎬 Initializing generation...")
        
        with torch.no_grad():
            # Set environment variable for better CUDA error reporting
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            
            progress_placeholder.info("🎬 Generating frames...")
            result = pipeline(**generation_kwargs)
            
            progress_placeholder.info("🎬 Processing frames...")
            
            # Get frames
            if hasattr(result, 'frames'):
                frames = result.frames[0]
            else:
                frames = result
            
            if not frames:
                raise Exception("No frames generated")
            
            # Debug frame information
            st.info(f"Generated {len(frames)} frames")
            if len(frames) > 0:
                first_frame = frames[0]
                if hasattr(first_frame, 'size'):
                    st.info(f"Frame format: PIL Image, size: {first_frame.size}")
                elif hasattr(first_frame, 'shape'):
                    st.info(f"Frame format: Array, shape: {first_frame.shape}")
                else:
                    st.info(f"Frame format: {type(first_frame)}")
            
            progress_placeholder.info(f"🎬 Converting {len(frames)} frames to video...")
            
            # Convert to video
            video_path = create_video_from_frames(
                frames, 
                params["width"], 
                params["height"], 
                params["fps"],
                progress_placeholder
            )
            
            progress_placeholder.empty()
            return video_path, None
            
    except Exception as e:
        return None, str(e)

def create_video_from_frames(frames, width, height, fps, progress_placeholder=None):
    """Create MP4 video from frames"""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        video_path = tmp.name
    
    # Create video using OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise Exception("Failed to open video writer")
    
    frame_count = 0
    for i, frame in enumerate(frames):
        try:
            # Convert frame to numpy array
            if hasattr(frame, 'size'):  # PIL Image
                frame_array = np.array(frame)
            elif hasattr(frame, 'numpy'):  # Torch tensor
                frame_array = frame.numpy()
            elif isinstance(frame, np.ndarray):  # Already numpy
                frame_array = frame
            else:
                frame_array = np.array(frame)
            
            # Ensure frame has the right shape (handle different formats)
            if len(frame_array.shape) == 4:  # Batch dimension
                frame_array = frame_array[0]
            elif len(frame_array.shape) == 2:  # Grayscale
                frame_array = np.stack([frame_array] * 3, axis=-1)
            
            # Ensure frame is the right size
            frame_height, frame_width = frame_array.shape[:2]
            if frame_height != height or frame_width != width:
                frame_pil = Image.fromarray(frame_array.astype(np.uint8))
                frame_pil = frame_pil.resize((width, height))
                frame_array = np.array(frame_pil)
            
            # Convert RGB to BGR for OpenCV
            if len(frame_array.shape) == 3 and frame_array.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame_array
            
            # Ensure frame is uint8
            if frame_bgr.dtype != np.uint8:
                if frame_bgr.max() <= 1.0:  # Normalized values
                    frame_bgr = (frame_bgr * 255).astype(np.uint8)
                else:  # Already in 0-255 range
                    frame_bgr = frame_bgr.astype(np.uint8)
            
            # Validate frame before writing
            if frame_bgr.shape[:2] != (height, width):
                st.warning(f"⚠️ Frame {i} has wrong dimensions: {frame_bgr.shape[:2]}, expected: ({height}, {width})")
                continue
            
            if len(frame_bgr.shape) != 3 or frame_bgr.shape[2] != 3:
                st.warning(f"⚠️ Frame {i} has wrong channel count: {frame_bgr.shape}")
                continue
            
            out.write(frame_bgr)
            frame_count += 1
            
            # Update progress
            if progress_placeholder and i % 5 == 0:  # Update every 5 frames
                progress_placeholder.info(f"🎬 Writing frame {i+1}/{len(frames)}...")
                
        except Exception as frame_error:
            st.warning(f"⚠️ Error processing frame {i}: {frame_error}")
            continue
    
    out.release()
    
    if frame_count == 0:
        raise Exception("No frames were successfully written to video")
    
    # Verify video file was created
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        raise Exception("Video file was not created or is empty")
    
    return video_path

def create_download_section(video_path, model_name, video_params):
    """Create download section for video"""
    st.subheader("📥 Download Video")
    
    # Video info
    file_size = os.path.getsize(video_path) / 1024 / 1024  # MB
    st.info(f"Video size: {file_size:.1f} MB")
    
    # Download button
    with open(video_path, 'rb') as f:
        video_data = f.read()
    
    filename = f"generated_video_{model_name.replace(' ', '_').replace(':', '_')}.mp4"
    
    st.download_button(
        label="📥 Download Video (MP4)",
        data=video_data,
        file_name=filename,
        mime="video/mp4",
        use_container_width=True
    )
    
    # Generation details
    with st.expander("📊 Generation Details"):
        st.json({
            "model": model_name,
            "prompt": video_params["prompt"],
            "frames": video_params["num_frames"],
            "fps": video_params["fps"],
            "duration": f"{video_params['num_frames'] / video_params['fps']:.1f}s",
            "resolution": f"{video_params['width']}x{video_params['height']}",
            "steps": video_params["steps"],
            "guidance_scale": video_params["guidance"],
            "seed": video_params["seed"] if video_params["seed"] is not None else "Random",
            "file_size": f"{file_size:.1f} MB"
        })

def main():
    """Main video generation interface"""
    st.title("🎬 AI Video Generator")
    st.markdown("Generate videos using state-of-the-art AI models")
    
    # Check dependencies
    missing_deps = []
    if not cv2:
        missing_deps.append("opencv-python")
    if not TextToVideoSDPipeline:
        missing_deps.append("diffusers[video]")
    
    if missing_deps:
        st.error("⚠️ Missing required dependencies for video generation:")
        for dep in missing_deps:
            st.error(f"- {dep}")
        
        st.info("Please install the required dependencies:")
        st.code(f"pip install {' '.join(missing_deps)}")
        st.stop()
    
    # Sidebar information
    with st.sidebar:
        st.header("🎬 Video Models")
        st.markdown("""
        **Available Models:**
        - **Zeroscope v2**: High quality, 576x320 resolution
        - **ModelScope T2V**: Good balance of quality and speed
        - **Text-to-Video Zero**: Experimental, image-based
        """)
        
        st.header("💡 Video Tips")
        st.markdown("""
        - Keep prompts simple and clear
        - Video generation takes much longer than images
        - Lower resolutions generate faster
        - Shorter videos (2-4 seconds) work best
        - Be patient - first generation downloads models
        """)
        
        # System requirements
        st.header("⚙️ System Status")
        if torch and torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory // 1024**3
            allocated = torch.cuda.memory_allocated(0) // 1024**3
            cached = torch.cuda.memory_reserved(0) // 1024**3
            free = vram - cached
            
            st.write(f"**GPU**: {torch.cuda.get_device_name(0)}")
            st.write(f"**Total VRAM**: {vram} GB")
            st.write(f"**Used**: {allocated} GB")
            st.write(f"**Cached**: {cached} GB")
            st.write(f"**Free**: {free} GB")
            
            if vram >= 12:
                st.success("✅ Excellent GPU for video generation")
            elif vram >= 8:
                st.info("ℹ️ Good GPU - use lower settings")
            elif vram >= 6:
                st.warning("⚠️ Limited VRAM - use very low settings")
            else:
                st.error("❌ Insufficient VRAM for video generation")
        else:
            st.error("❌ GPU strongly recommended for video generation")
            if not torch:
                st.error("PyTorch not available")
        
        # Memory management
        st.header("🧹 Memory Management")
        if st.button("🗑️ Clear GPU Cache"):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                st.success("GPU cache cleared!")
                st.rerun()
        
        if st.button("🔄 Reset Models"):
            st.cache_resource.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            st.success("All models reset!")
            st.rerun()
        
        # GPU test
        if st.button("🔧 Test GPU"):
            if torch and torch.cuda.is_available():
                try:
                    test_tensor = torch.randn(100, 100, device='cuda')
                    result = torch.matmul(test_tensor, test_tensor)
                    del test_tensor, result
                    torch.cuda.synchronize()
                    st.success("✅ GPU test passed!")
                except Exception as e:
                    st.error(f"❌ GPU test failed: {e}")
            else:
                st.error("❌ No CUDA GPU available")

    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Model selection
        selected_model_name = st.selectbox(
            "🤖 Choose Video Model:",
            options=list(AVAILABLE_VIDEO_MODELS.keys()),
            index=0,
            help="Different models have different strengths and speeds"
        )
        
        # Prompt input
        prompt = st.text_area(
            "✍️ Describe your video:",
            placeholder="A cat playing with a ball of yarn in slow motion...",
            height=100,
            help="Keep it simple and descriptive for best results"
        )
        
        # Quick prompts
        st.markdown("**🎬 Quick Prompts:**")
        prompt_col1, prompt_col2 = st.columns(2)
        with prompt_col1:
            if st.button("🌊 Ocean Waves"):
                prompt = "Ocean waves crashing on a sandy beach at sunset"
            if st.button("🔥 Campfire"):
                prompt = "A cozy campfire with dancing flames at night"
        with prompt_col2:
            if st.button("🌸 Flower Bloom"):
                prompt = "A flower blooming in time-lapse photography"
            if st.button("☁️ Clouds"):
                prompt = "Fluffy white clouds moving across a blue sky"
    
    with col2:
        st.markdown("### ⚙️ Video Settings")
        
        # Recommended settings based on GPU memory
        if torch and torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory // 1024**3
            if vram >= 12:
                default_frames = 24
                max_frames = 64
                default_steps = 20
            elif vram >= 8:
                default_frames = 16
                max_frames = 32
                default_steps = 15
                st.info("💡 Using reduced settings for 8GB GPU")
            elif vram >= 6:
                default_frames = 8
                max_frames = 16
                default_steps = 10
                st.warning("⚠️ Using minimal settings for 6GB GPU")
            else:
                default_frames = 8
                max_frames = 8
                default_steps = 10
                st.error("❌ Very limited settings for <6GB GPU")
        else:
            default_frames = 8
            max_frames = 16
            default_steps = 10
        
        # Video parameters
        num_frames = st.slider(
            "🎞️ Number of Frames",
            min_value=8, max_value=max_frames, value=default_frames,
            help="More frames = longer video but slower generation"
        )
        
        fps = st.slider(
            "📹 Frames Per Second",
            min_value=4, max_value=30, value=8,
            help="Higher FPS = smoother motion"
        )
        
        duration = round(num_frames / fps, 1)
        st.info(f"Video duration: ~{duration} seconds")
    
    # Advanced settings
    with st.expander("🔧 Advanced Video Settings"):
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            steps = st.slider(
                "Inference Steps",
                min_value=10, max_value=100, value=default_steps,
                help="More steps = better quality but much slower"
            )
            
            guidance = st.slider(
                "Guidance Scale",
                min_value=1.0, max_value=20.0, value=9.0, step=0.5,
                help="Higher values follow prompt more closely"
            )
        
        with col_adv2:
            use_seed = st.checkbox("🎲 Use Fixed Seed")
            seed = None
            if use_seed:
                seed = st.number_input(
                    "Seed Value",
                    min_value=0, max_value=2147483647, value=42
                )
            
            # Recommended resolutions based on GPU
            if torch and torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_memory // 1024**3
                if vram >= 12:
                    height_options = [320, 576, 768]
                    width_options = [576, 1024, 1024]
                    default_h_idx = 1
                    default_w_idx = 0
                elif vram >= 8:
                    height_options = [320, 576]
                    width_options = [576, 768]
                    default_h_idx = 0
                    default_w_idx = 0
                else:
                    height_options = [320]
                    width_options = [576]
                    default_h_idx = 0
                    default_w_idx = 0
            else:
                height_options = [320]
                width_options = [576]
                default_h_idx = 0
                default_w_idx = 0
                
            height = st.selectbox("Height", height_options, index=default_h_idx)
            width = st.selectbox("Width", width_options, index=default_w_idx)
            
            # Memory usage estimate
            memory_estimate = (height * width * num_frames * steps) / 1024**3 * 2  # Rough estimate
            st.info(f"Est. memory: ~{memory_estimate:.1f}GB")
    
    # Generate button
    if st.button("🎬 Generate Video", type="primary", use_container_width=True):
        if not prompt.strip():
            st.warning("⚠️ Please enter a prompt to generate video.")
            return
        
        # Prepare parameters
        params = {
            "num_frames": num_frames,
            "steps": steps,
            "guidance": guidance,
            "height": height,
            "width": width,
            "fps": fps,
            "seed": seed,
            "prompt": prompt
        }
        
        # Load model
        selected_model_id = AVAILABLE_VIDEO_MODELS[selected_model_name]
        
        with st.spinner(f"🔄 Loading {selected_model_name}..."):
            # Determine loading strategy
            if torch and torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_memory // 1024**3
                if vram >= 8:
                    device_strategy = "cuda"
                    dtype = torch.float16
                else:
                    device_strategy = "cpu_offload"
                    dtype = torch.float16
            else:
                device_strategy = "cpu"
                dtype = torch.float32
            
            pipeline = load_video_model(selected_model_id, device_strategy, dtype)
        
        if pipeline is None:
            st.error("❌ Failed to load the selected model. Please try a different model.")
            return
        
        # Generate video
        with st.spinner(f"🎬 Generating video ({duration}s, {num_frames} frames)..."):
            video_path, error = generate_video(pipeline, prompt, params)
        
        if video_path:
            st.success(f"✅ Video generated successfully!")
            
            # Display video
            st.video(video_path)
            
            # Download section
            create_download_section(video_path, selected_model_name, params)
            
            # Clean up
            try:
                os.unlink(video_path)
            except:
                pass
        else:
            st.error(f"❌ Video generation failed: {error}")
            
            # Show specific error help
            if "CUDA" in str(error) and "out of memory" in str(error):
                st.error("**Try these solutions:**")
                st.error("- Reduce number of frames")
                st.error("- Lower resolution")
                st.error("- Reduce inference steps")
                st.error("- Close other GPU applications")
            elif "CUDA" in str(error):
                st.error("**CUDA-specific solutions:**")
                st.error("- Restart the application")
                st.error("- Update GPU drivers")
                st.error("- Check GPU status")


# Run standalone if executed directly
if __name__ == "__main__":
    main()

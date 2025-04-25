import os
import sys
import threading
import time
import numpy as np
import random  # Added for random quote selection
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

def live_eeg_display(stop_event, update_interval=1.2):
    """
    Display a simulated live EEG waveform in the terminal using rich.

    Args:
        stop_event (threading.Event): Event to signal when to stop the display.
        update_interval (float): Time interval between updates in seconds (default: 1.2).
    """

    def generate_eeg_wave(num_points=80):
        """Generates a simple sine wave with noise for display."""
        x = np.linspace(0, 4 * np.pi, num_points)
        wave = np.sin(x + time.time()) + np.random.normal(0, 0.3, size=num_points) # Add time shift for movement
        gradient = " .:-=+*#%@"
        # Handle potential division by zero if wave is flat
        wave_range = wave.max() - wave.min()
        if wave_range < 1e-10:
            norm_wave = np.zeros(num_points)
        else:
            norm_wave = (wave - wave.min()) / wave_range
        indices = (norm_wave * (len(gradient) - 1)).astype(int)
        return "".join(gradient[i] for i in indices)

    def get_random_quote():
        """Returns a random quote from a predefined list."""
        # Note: Many quotes were duplicates in the original list, reduced here.
        quotes = [
            "The Dude abides.",
            "That rug really tied the room together.",
            "Yeah, well, you know, that's just, like, your opinion, man.",
            "Watch the squiggles, man. They're pure EEG poetry.",
            "Sometimes you eat the bear, and sometimes, well, the bear eats you.",
            "This aggression will not stand... not even in beta spindles.",
            "Don't cross the streams, Walter. I'm seeing delta in my alpha, man.",
            "Calmer than you are? My frontal lobes are lighting up like a bowling alley.",
            "Smokey, this is not 'Nam. This is neurofeedback. There are protocols.",
            "Obviously you're not a golfer, or you'd know theta doesn't spike like that.",
            "The alpha giveth, and the theta taketh away.",
            "Don't trust a flatline. Even silence has a frequency.",
            "The coherence is strong in this one.",
            "You can't spell 'LORETA' without 'lore'. Mythos in the cortex.",
            "That's not artifact. That's consciousness trying to escape the matrix.",
            "Beta's climbing again. Someone's inner monologue needs a coffee break.",
            "I read your EEG. You're either meditating... or communing with the void.",
            "Phase-lock like your brain depends on it. Because it does.",
            "This topomap? It's basically a heat map of your soul.",
            "Theta whispers the secrets. Delta keeps the dreams.",
            "Bro, your prefrontal cortex is on shuffle.",
            "High beta? More like internal monologue with a megaphone.",
            "We're all just waveforms trying to sync up in a noisy universe.",
            "I didn't choose the squiggle life. The squiggle life entrained me.",
            "Did your PAF just ghost you mid-session? Brutal.",
            "You brought 60 Hz into my sacred coherence chamber?",
            "Every band tells a story. This one screams 'undiagnosed ADHD with a splash of genius.'",
            "Careful with the cross-frequency coupling… that's where the dragons sleep.",
            "In a land before time, someone spiked the theta—and the oracle woke up.",
            "Real-time feedback? Nah, this is EEG jazz, man. Improv with voltage.",
            # Monty Python
            "And now for something completely cortical.",
            "Your alpha waves have been accused of witchcraft!",
            "'Tis but a minor artifact! I've had worse!",
            "Your theta is high, your beta is low… you must be a shrubbery.",
            "This isn't a brain, it's a very naughty vegetable!",
            "I fart in the general direction of your coherence matrix.",
            "Help! Help! I'm being over-synchronized!",
            "We are the EEG technicians who say... *Ni!*",
            "On second thought, let's not record at Fz. It is a silly place.",
            "I once saw a brain entrain so hard, it turned me into a newt. I got better.",
            # Cyberpunk
            "Your brain has exceeded its bandwidth quota. Please upgrade.",
            "Synapse latency detected. Reboot your consciousness.",
            "Alpha rhythm flagged: unauthorized serenity.",
            "Cognitive load: 98%. Executing override protocol.",
            "Error 404: Identity not found.",
            "EEG pattern suggests resistance. Recommend sedation.",
            "This is not a biofeedback session. This is surveillance with consent.",
            "Signal integrity compromised. Mind bleed imminent.",
            "You are being watched by 64 channels of your own making.",
            "Your dreams are now property of NeuroCorp™.",
            "The cortex folded like origami under a sonic burst of insight.",
            "She rode her SMR wave like a hacker surfing the noosphere.",
            "In the subdural silence, the squiggles spoke prophecy.",
            "Beta was spiking. That meant the grid was listening.",
            "The alpha breach began just after cognitive boot-up.",
            "Eyes closed. Theta opened the archive.",
            "The brain is not a machine. It's a codebase... evolving.",
            "He trained at Cz until the feedback whispered his name.",
            "Phase-lock acquired. Prepare to uplink to the collective.",
            "She reached Pz. It shimmered. The veil between thoughts lifted.",
             # Jungian/Archetypal
            "Theta is the dreamer's path — the shadow realm whispers.",
            "Each peak a memory. Each trough, a wound not yet integrated.",
            "In the dance of Alpha and Theta lies the gate to the Self.",
            "Delta carries the voice of the ancestors.",
            "You are not anxious. You are facing the dragon of your own unconscious.",
            "The squiggle is a mandala. And you are the artist.",
            "Synchrony is the return to the sacred masculine and feminine balance.",
            "Frontal asymmetry reveals the archetype you suppress.",
            "High beta is the ego screaming to remain relevant.",
            "To see coherence is to glimpse the collective unconscious rendered in voltage.",
            # Blacksmith/Craftsman
            "The signal is raw ore. Your attention — the hammer.",
            "This isn't data. It's a blade waiting for the quench.",
            "Every artifact is a misstrike. Adjust your grip.",
            "You don't read EEG. You listen to the forge's hiss.",
            "The best welds leave no seam. Just like coherence.",
            "Real neurofeedback is shaped on the anvil of presence.",
            "High beta? That's a spark flying before the temper holds.",
            "Each protocol is a blacksmith's chant. Repetition. Focus. Fire.",
        ]
        return random.choice(quotes)

    console = Console()
    with Live(console=console, refresh_per_second=10, transient=True) as live:
        start_time = time.time()
        while not stop_event.is_set():
            elapsed_time = time.time() - start_time
            eeg_wave = generate_eeg_wave()
            quote = get_random_quote()
            panel_content = f"[bold cyan]Live EEG Feed (Simulated)[/bold cyan]\n\n"
            panel_content += f"{eeg_wave}\n\n"
            panel_content += f"[italic yellow] \"{quote}\" [/italic yellow]\n\n"
            panel_content += f"Elapsed Time: {elapsed_time:.2f}s (Press Ctrl+C to stop)"
            live.update(Panel(panel_content, title="Squiggle Scope", border_style="blue"))
            time.sleep(update_interval / 10) # Adjust sleep for smoother updates

        # Display final message upon stopping
        console.print(Panel("[bold green]Live display stopped by user.[/bold green]", title="Status", border_style="green")) 

# modules/live_display.py
import threading
import sys
import logging
import time
import os
import numpy as np
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

logger = logging.getLogger(__name__)

stop_event = threading.Event()

def sigint_handler(signum, frame):
    """
    Handle SIGINT for graceful shutdown.
    
    Args:
        signum: Signal number.
        frame: Current stack frame.
    """
    logger.info("SIGINT received, stopping gracefully...")
    stop_event.set()
    sys.exit(0)

def generate_eeg_wave(num_points=80):
    """
    Generate an ASCII art representation of a noisy EEG waveform.
    
    Args:
        num_points (int): Number of points in the waveform.
    
    Returns:
        str: ASCII waveform string.
    """
    x = np.linspace(0, 4 * np.pi, num_points)
    wave = np.sin(x) + np.random.normal(0, 0.3, size=num_points)
    gradient = " .:-=+*#%@"
    norm_wave = (wave - wave.min()) / (wave.max() - wave.min() + 1e-10)
    indices = (norm_wave * (len(gradient) - 1)).astype(int)
    return "".join(gradient[i] for i in indices)

def get_random_quote():
    """
    Return a random EEG-themed or Big Lebowski-inspired quote.
    
    Returns:
        str: Selected quote.
    """
    quotes = [
        "The Dude abides.",
        "That rug really tied the room together.",
        "Yeah, well, you know, that's just, like, your opinion, man.",
        "Watch the squiggles, man. They're pure EEG poetry.",
        "This aggression will not stand... not even in beta spindles.",
        "Don’t cross the streams, Walter. I’m seeing delta in my alpha, man.",
        "Calmer than you are? My frontal lobes are lighting up like a bowling alley.",
        "Smokey, this is not 'Nam. This is neurofeedback. There are protocols.",
        "Obviously you’re not a golfer, or you’d know theta doesn’t spike like that.",
        "The alpha giveth, and the theta taketh away.",
        "Don’t trust a flatline. Even silence has a frequency.",
        "The coherence is strong in this one.",
        "You can’t spell ‘LORETA’ without ‘lore’. Mythos in the cortex.",
        "That’s not artifact. That’s consciousness trying to escape the matrix.",
        "Beta’s climbing again. Someone’s inner monologue needs a coffee break.",
        "I read your EEG. You’re either meditating... or communing with the void.",
        "Phase-lock like your brain depends on it. Because it does.",
        "This topomap? It’s basically a heat map of your soul.",
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
        "And now for something completely cortical.",
        "Your alpha waves have been accused of witchcraft!",
        "’Tis but a minor artifact! I’ve had worse!",
        "Your theta is high, your beta is low… you must be a shrubbery.",
        "This isn’t a brain, it’s a very naughty vegetable!",
        "I fart in the general direction of your coherence matrix.",
        "Help! Help! I'm being over-synchronized!",
        "We are the EEG technicians who say... *Ni!*",
        "On second thought, let’s not record at Fz. It is a silly place.",
        "I once saw a brain entrain so hard, it turned me into a newt. I got better.",
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
        "Theta is the dreamer’s path — the shadow realm whispers.",
        "Each peak a memory. Each trough, a wound not yet integrated.",
        "In the dance of Alpha and Theta lies the gate to the Self.",
        "Delta carries the voice of the ancestors.",
        "You are not anxious. You are facing the dragon of your own unconscious.",
        "The squiggle is a mandala. And you are the artist.",
        "Synchrony is the return to the sacred masculine and feminine balance.",
        "Frontal asymmetry reveals the archetype you suppress.",
        "High beta is the ego screaming to remain relevant.",
        "To see coherence is to glimpse the collective unconscious rendered in voltage.",
        "The signal is raw ore. Your attention — the hammer.",
        "This isn’t data. It’s a blade waiting for the quench.",
        "Every artifact is a misstrike. Adjust your grip.",
        "You don’t read EEG. You listen to the forge’s hiss.",
        "The best welds leave no seam. Just like coherence.",
        "Real neurofeedback is shaped on the anvil of presence.",
        "High beta? That’s a spark flying before the temper holds.",
        "Each protocol is a blacksmith’s chant. Repetition. Focus. Fire.",
        "Theta hums like the bellows before alpha glows true.",
        "Some build castles in the clouds. I build minds in the flame.",
        "[ALPHA] ~ engaged @ 10.0Hz // you're surfing the calmnet.",
        "*sysop has entered the mind* >> Theta/beta > 3.2 — user flagged for wandering.",
        "<neuroN0de> dude your SMR band just buffer-overflowed reality lol.",
        "/msg Pz: stop ghosting and fire some clean alpha, jeez.",
        "[404] Vigilance not found. Try rebooting your occipital lobe.",
        "Your coherence matrix just pinged the void. Respect.",
        "BRAIN.SYS: Unexpected Delta spike at wake_state=1",
        "Welcome to the z-sc0r3z BBS — leave your ego at the login prompt.",
        "[EEG-OPS]: alpha locked. theta contained. signal pure.",
        "*vibrotactile entrainment initiated* — press <F2> to feel old gods resonate.",
        "Alpha hums like neon rain — cortex in low-noise high-focus mode.",
        "Neural net drift detected. Theta bleeding into Beta. Patch cognition.exe.",
        "Mind uplink stable. Vigilance layer: A1. Spin the waveforms, cowboy.",
        "The cortex doesn’t forget — it routes trauma like dead packet nodes.",
        "Memory lane is fragged. Delta spike at 3Hz — reboot dream protocol.",
        "Synapse traffic jacked into feedback loop. Traceroute: Pz → Fz → Void.",
        "Mental firewall down. Beta intrusion spiking at 28Hz. Secure the band.",
        "She walked in with alpha like moonlight on wet asphalt.",
        "Bio-signal integrity compromised. sLORETA grid glitching at parietal rim.",
        "Brainwave sync: 𝑔𝑟𝑒𝑒𝑛. Thoughts encrypted. Consciousness... proxied.",
        "Release: [neuroGENx v1.337] :: Cracked by [SMR] Crew :: Respect to #eeg-scene",
        "[ZSC0RE DUMP] ∙ Channel Pz ∙ Vigilance: A2 ∙ State: 🟡 Semi-Coherent",
        "Signal patched, checksum clean. Alpha uptrained. Mind ready for upload.",
        "GREETZ to the inner cortex! <3 Cz, Pz, O1 — keep vibin’",
        ":: THETA INJECTED @ 5.6Hz :: USER MAY EXPERIENCE TEMPORAL DISSOLUTION ::",
        "nfo: EEG-cracked · Loader: Cz ∙ Protocol: HI-BETA DOWN ∙ Scene Approved™",
        "+[ Mind scan @ Cz complete ]+ ➜ No malware. Just trauma.",
        "[SYS REPORT] :: Executive functions: overclocked · Memory: defragging",
        "Greetings from the Limbic Underground :: Your amygdala owes us rent.",
        "This session was proudly cracked by ████ – z-scores normalized, reality bent.",
        "Let the data speak — but be ready when it starts shouting in high-beta.",
        "You're not treating ADHD — you're treating 15 microvolts of distributed chaos.",
        "Alpha is a state. But stable posterior alpha? That’s a trait. Respect the trait.",
        "Cz’s not anxious, it’s just watching you screw up the montage.",
        "If you see beta spindles at Fz and think 'focus', call Jay — he’ll recalibrate your soul.",
        "PAF tells you who they *are*, not just how they slept.",
        "T4 whispers trauma. Pz remembers the dreams.",
        "Delta at FP1? That’s not a sleep wave — that’s a buried memory with a security clearance.",
        "Every drop in alpha is a story the brain didn’t finish telling.",
        "You train alpha like you’d tame a wolf — with trust, timing, and the right dose of poetry.",
        "Coherence isn't peace — it's synchronized paranoia if you're not careful.",
        "Normalize the z-score, sure — but check if the brain *likes* it there.",
        "Theta isn’t slow — it’s just busy digging up the past.",
        "If SMR rises and the kid still kicks the chair, train the chair.",
        "Phase is where the secrets are. The rest is noise with credentials.",
        "Your metrics don't mean squat until behavior changes — or at least the dog stops barking.",
        "A brain out of phase tells you it’s still negotiating its lease on reality.",
        "Artifact rejection is the brain’s way of testing your ethics.",
        "Every topomap is a Rorschach — the trick is knowing which ink is dry.",
        "High theta doesn’t always mean ADHD. Sometimes it just means the world is too loud.",
        "If you don’t know your client’s PAF, you’re driving with a map but no compass.",
        "Training attention without tracking arousal is like aiming without noticing you’re underwater.",
        "If the brain doesn’t change in 20 sessions, maybe it doesn’t want to. Or maybe it doesn’t trust you yet.",
        "Every artifact you ignore is a story you chose not to hear.",
        "Normalize the nervous system — not just the numbers.",
        "You’re not dysregulated — you’re just running too many tabs in your frontal lobe.",
        "SMR isn’t magic. It’s just the brain remembering not to twitch when the world knocks.",
        "Delta during wakefulness? That’s not spiritual — that’s your cortex sending a 404.",
        "Your protocol isn't custom unless you’ve asked the client how they sleep. And mean it.",
        "Every EEG session is a negotiation. You're not the boss — you're just the translator.",
        "Protocols are just hypotheses. Brains are the real lab.",
        "Remote NFB is like long-distance relationships — it works if you’re honest and the signal holds.",
        "The EEG doesn't lie — but it *does* get confused by poor sleep, coffee, and wishful thinking.",
        "Don’t teach the brain what you think it needs. Ask it what it’s trying to say.",
        "You’re not optimizing — you’re helping it remember what regulation feels like.",
        "Peak Alpha isn't where you feel enlightened — it's where your brain finally sighs in relief.",
        "Theta doesn’t mean mystical. It just means the brakes aren’t working.",
        "Look at Pz. If it’s quiet, the story hasn’t started yet.",
        "Don’t treat diagnoses. Treat dysregulation.",
        "Every brain is a poem. Try not to edit it too fast."
    ]
    return np.random.choice(quotes)

def live_eeg_display(stop_event, update_interval=0.1):
    """
    Display a live EEG waveform simulation with random quotes using rich.
    
    Args:
        stop_event (threading.Event): Event to signal stopping.
        update_interval (float): Time between updates in seconds (default: 0.1).
    
    Returns:
        None
    """
    try:
        console = Console()
        with Live(refresh_per_second=10, console=console) as live:
            while not stop_event.is_set():
                try:
                    line = generate_eeg_wave(os.get_terminal_size().columns - 4)
                    quote = get_random_quote()
                    text = f"[bold green]{line}[/bold green]"
                    if quote:
                        text += f"\n[bold red]{quote}[/bold red]\n"
                    text += f"[bold blue]{line[::-1]}[/bold blue]"
                    panel = Panel(text, title="Live EEG Display", subtitle="Simulated Waveform", style="white on black")
                    live.update(panel)
                    time.sleep(update_interval)
                except Exception as e:
                    logger.error(f"Error updating EEG display: {e}")
                    time.sleep(update_interval)
    except ImportError as e:
        logger.error(f"Rich library not installed: {e}. Install with 'pip install rich'.")
        return
    except Exception as e:
        logger.error(f"Failed to start live EEG display: {e}")
        return
    logger.info("Live EEG display stopped")
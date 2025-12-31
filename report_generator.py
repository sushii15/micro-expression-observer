from detector import MicroEvent
import time

class ReportGenerator:
    def __init__(self, style="plain"):
        self.style = style # 'plain' or 'technical'
        
    def generate(self, events, session_duration):
        lines = []
        lines.append("MICRO-EXPRESSION OBSERVATION REPORT")
        lines.append("===================================")
        lines.append(f"Date: {time.ctime()}")
        lines.append(f"Duration: {session_duration:.1f} seconds")
        lines.append(f"Events Detected: {len(events)}")
        lines.append("-" * 40)
        lines.append("")
        
        if not events:
            lines.append("No significant facial micro-movements detected exceeding baseline thresholds.")
            return "\n".join(lines)
            
        # Sort by time
        events.sort(key=lambda x: x.start_time)
        
        if self.style == "plain":
            lines.append(self._generate_plain(events))
        else:
            lines.append(self._generate_technical(events))
            
        return "\n".join(lines)

    def _generate_plain(self, events):
        buffer = []
        for e in events:
            # Map AU to plain English
            desc = "unknown movement"
            if e.au_type == "AU01": desc = "brief eyebrow raise"
            elif e.au_type == "AU04": desc = "brow lowering"
            elif e.au_type == "AU06": desc = "eye narrowing"
            elif e.au_type == "AU12": desc = "lip corner pull"
            elif e.au_type == "AU15": desc = "lip corner depression"
            
            intensity = "slight"
            if e.intensity_z > 3.0: intensity = "distinct"
            
            buffer.append(f"At {e.start_time:.1f}s, a {intensity} {desc} was observed (duration: {e.duration*1000:.0f}ms). Predicted Context: {e.emotion_label}.")
            
        return "\n".join(buffer)

    def _generate_technical(self, events):
        buffer = []
        buffer.append(f"{'Time':<10} | {'AU Code':<10} | {'Intensity (Z)':<15} | {'Emotion':<12} | {'Duration'}")
        buffer.append("-" * 75)
        
        for e in events:
            buffer.append(f"{e.start_time:<10.2f} | {e.au_type:<10} | {e.intensity_z:<15.2f} | {e.emotion_label:<12} | {e.duration:.3f}s")
            
        return "\n".join(buffer)

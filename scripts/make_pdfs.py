"""Write the corpus PDFs. Run once: uv run --group corpus python scripts/make_pdfs.py"""
from pathlib import Path

from fpdf import FPDF

PDFS = {
    "corpus/hr/code-of-conduct.pdf": ("Code of Conduct", [
        ("Our standards", "Treat colleagues, customers and partners with respect. Harassment and discrimination are not tolerated."),
        ("Conflicts of interest", "Disclose any outside role, investment or relationship that could influence your work for Brackwater. Gifts worth more than $100 must be declared to Legal."),
        ("Safety", "Never disable a robot safety system, including emergency stops and speed limits, outside an approved test."),
        ("Speaking up", "Report concerns to your manager, People Operations or the anonymous ethics line. Brackwater does not tolerate retaliation."),
    ]),
    "corpus/it/acceptable-use-policy.pdf": ("Acceptable Use Policy", [
        ("Company devices", "Company laptops are for Brackwater work. Limited personal use is allowed if it does not affect security or performance."),
        ("Software", "Install software only from Self Service, or after a security review."),
        ("Customer data", "Customer robot telemetry may only be copied to systems approved for Confidential data. Never paste customer data into public AI tools."),
        ("Monitoring", "Brackwater monitors company devices and accounts for security purposes."),
    ]),
    "corpus/product/tern-datasheet-v1.pdf": ("Tern AMR Datasheet (Gen 1)", [
        ("Payload", "Up to 500 kg."),
        ("Battery", "8 hours of typical operation; full charge in 90 minutes."),
        ("Speed", "Up to 1.8 m/s in robot-only zones."),
        ("Navigation", "2D lidar with natural-feature navigation; no floor markers needed."),
    ]),
    "corpus/product/tern-datasheet-v2.pdf": ("Tern AMR Datasheet (Gen 2)", [
        ("Payload", "Up to 600 kg."),
        ("Battery", "9 hours of typical operation; 80% charge in 50 minutes."),
        ("Speed", "Up to 2.0 m/s in robot-only zones."),
        ("Navigation", "3D lidar and stereo cameras with natural-feature navigation."),
    ]),
    "corpus/product/heron-datasheet.pdf": ("Heron Picking Arm Datasheet", [
        ("Reach", "1.4 m."),
        ("Payload", "Up to 12 kg per pick."),
        ("Throughput", "Up to 600 picks per hour for boxed goods."),
        ("Mounting", "Mounts on a Tern Gen 2 or on a fixed station."),
    ]),
    "corpus/hr/handbook-summary-2025.pdf": ("Employee Handbook 2025: Summary Slides", [
        ("Time off", "25 vacation days a year; carry over up to 5 days."),
        ("Where you work", "Two office days a week, agreed with your team."),
        ("Parental leave", "12 weeks paid, after probation."),
        ("Questions", "Ask People Operations or see the full handbook."),
    ]),
}

for path, (title, sections) in PDFS.items():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 10, f"Brackwater Robotics - {title}", new_x="LMARGIN", new_y="NEXT")
    for heading, body in sections:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 8, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, body, new_x="LMARGIN", new_y="NEXT")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pdf.output(path)
    print(path)

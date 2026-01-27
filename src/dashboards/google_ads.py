import sys
from pathlib import Path
import streamlit as st
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
CAMPAIGNS_DIR = ROOT / "campaigns"
if str(CAMPAIGNS_DIR) not in sys.path:
    sys.path.append(str(CAMPAIGNS_DIR))

from campaigns.streamlit_campaigns import StreamlitGADS


def main():
    dashboard = StreamlitGADS()
    dashboard.main_dash()
    dashboard.general_graph()
    dashboard.cluster()
    dashboard.cluster_barplot()
    dashboard.perf_barplot()

if __name__ == "__main__":
    main()

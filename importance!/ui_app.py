import os
import tempfile
from typing import Any, Dict

import streamlit as st

from multimodal_rag_observation_basic import MultimodalRAGSystemTSDRIB


st.set_page_config(page_title="TSD RIB Multimodal RAG", layout="wide")


def build_db_config() -> Dict[str, Any]:
    return {
        "host": st.sidebar.text_input("DB Host", value="localhost"),
        "database": st.sidebar.text_input("DB Name", value="tsdsrd"),
        "user": st.sidebar.text_input("DB User", value="amir"),
        "password": st.sidebar.text_input("DB Password", value="amir123", type="password"),
        "port": st.sidebar.number_input("DB Port", value=5432, step=1),
    }


@st.cache_resource(show_spinner=False)
def get_rag_system(db_config: Dict[str, Any], vllm_url: str, owlv2_url: str, max_image_size: int):
    return MultimodalRAGSystemTSDRIB(
        db_config=db_config,
        vllm_url=vllm_url,
        owlv2_url=owlv2_url,
        max_image_size=max_image_size,
    )


st.title("TSD RIB Multimodal RAG UI")

with st.sidebar:
    st.header("Server Settings")
    vllm_url = st.text_input("VLLM URL", value="http://localhost:8000")
    owlv2_url = st.text_input("OWLv2 URL", value="http://localhost:8010")
    max_image_size = st.slider("Max Image Size", min_value=512, max_value=2048, value=1536, step=128)

    st.header("Database Settings")
    db_config = build_db_config()

    st.header("Query Settings")
    top_k = st.slider("Top K", min_value=1, max_value=10, value=3, step=1)
    similarity_threshold = st.slider("Similarity Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    use_object_detection = st.checkbox("Use OWLv2 Object Detection", value=True)

st.subheader("Text Query")
text_query = st.text_area("Enter a text query", height=120)

st.subheader("Image Query")
image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])

run_button = st.button("Run Query", type="primary")

if run_button:
    if not text_query and image_file is None:
        st.warning("Please provide either a text query or an image.")
        st.stop()

    with st.spinner("Initializing RAG system..."):
        rag = get_rag_system(db_config, vllm_url, owlv2_url, max_image_size)

    if image_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{image_file.name}") as tmp:
            tmp.write(image_file.getbuffer())
            image_path = tmp.name

        st.image(image_file, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Running image pipeline..."):
            result = rag.query_with_image(
                image_path=image_path,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                temperature=temperature,
                show_sources=True,
                use_object_detection=use_object_detection,
            )

        os.unlink(image_path)
    else:
        with st.spinner("Running text pipeline..."):
            result = rag.query_with_text(
                query_text=text_query,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                temperature=temperature,
                show_sources=True,
            )

    if not result.get("success"):
        st.error(result.get("error", "Unknown error"))
        st.stop()

    if result.get("image_description"):
        st.markdown("### Image Description")
        st.write(result["image_description"])

    if result.get("semantic_top_match"):
        top = result["semantic_top_match"]
        st.markdown("### Top Semantic Match (Text-Based Search)")
        st.info(f"📊 {top['section']} - {top['title']} (Similarity: {top['similarity']})")
        st.caption("⚠️ This is the top text similarity match. The LLM may select a different section after visual analysis.")

    # Extract the LLM's selected section from the response
    answer_text = result.get("answer", "")
    selected_section = None
    if answer_text:
        # Look for "SELECTED RIB SECTION: X.X - Title" pattern
        import re
        match = re.search(r'SELECTED RIB SECTION:\s*([\d.]+)\s*-\s*([^\n]+)', answer_text, re.IGNORECASE)
        if match:
            selected_section = f"{match.group(1)} - {match.group(2).strip()}"
    
    if selected_section:
        st.markdown("### ✅ Final Selected RIB Section (LLM Decision)")
        st.success(f"🎯 {selected_section}")
        st.caption("This is the LLM's final decision after analyzing the image, OWLv2 detections, and context.")
    
    st.markdown("### Detailed Recommendations")
    st.write(answer_text)

    if result.get("owlv2_detection"):
        owlv2 = result["owlv2_detection"]
        st.markdown("### OWLv2 Detection Results")
        cols = st.columns(3)
        cols[0].metric("Tier 1 Detections", owlv2.get("tier1_detection_count", 0))
        cols[1].metric("Primary Subsection", owlv2.get("primary_rib_subsection", "-") or "-")
        cols[2].metric("Categories", ", ".join(owlv2.get("broad_categories", [])) or "-")

        top5 = owlv2.get("top5_rib_subsections", [])
        if top5:
            st.markdown("#### 🏆 Top 5 RIB Subsections (Object Detection)")
            st.caption("Ranked by detection confidence and count from OWLv2 visual analysis")
            st.table([
                {
                    "Rank": sub.get("rank"),
                    "Section": sub.get("section"),
                    "Score": round(sub.get("score", 0), 3),
                    "Detections": sub.get("detection_count", 0),
                }
                for sub in top5
            ])
        else:
            st.info("No RIB subsections detected in the image")

    if result.get("sources"):
        st.markdown("### Sources")
        st.table([
            {
                "Section": src.get("section"),
                "Title": src.get("title"),
                "Category": src.get("category"),
                "Similarity": src.get("similarity"),
            }
            for src in result["sources"]
        ])

    st.markdown("### Performance")
    st.json(
        {
            "total_time": result.get("total_time", 0),
            "tokens_per_second": result.get("tokens_per_second", 0),
            "num_sources": result.get("num_sources", 0),
        }
    )

import streamlit as st
from PIL import Image
from calorie_graph import calorie_graph

st.title("🥗 Food Calories and Proteins Analyzer")

uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Food Image", width="stretch")

    # Convert to bytes
    uploaded_file.seek(0)
    image_bytes = uploaded_file.read()
    mime = uploaded_file.type

    # ✅ Invoke LangGraph pipeline
    with st.spinner("Analyzing..."):
        result_state = calorie_graph.invoke({
            "image_bytes": image_bytes,
            "mime": mime
        })

    st.subheader("Nutrition Analysis")
    st.write(result_state["result"])

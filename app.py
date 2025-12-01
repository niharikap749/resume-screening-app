import streamlit as st
import pickle
import re
import nltk

# optional: PDF / DOCX reading
import PyPDF2
# import docx  # uncomment if you want to support .docx files

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# ----- load models -----
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))


def cleanResume(resume_txt: str) -> str:
    """Basic cleaning for resume text (returns a single cleaned string)."""
    if not isinstance(resume_txt, str):
        resume_txt = str(resume_txt)

    txt = resume_txt
    txt = re.sub(r'http\S+', ' ', txt)                       # URLs
    txt = re.sub(r'RT|cc', ' ', txt)                         # RT / cc
    txt = re.sub(r'#\S+', ' ', txt)                          # hashtags
    txt = re.sub(r'@\S+', ' ', txt)                          # mentions / emails (partial)
    txt = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', txt)
    txt = re.sub(r'[^\x00-\x7f ]', ' ', txt)                 # non-ascii
    txt = re.sub(r'\s+', ' ', txt)                           # repeated whitespace
    return txt.strip()


def extract_text_from_pdf(uploaded_file) -> str:
    """Read uploaded PDF bytes (Streamlit UploadedFile) and extract text."""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text_pages = []
        for p in reader.pages:
            page_text = p.extract_text()
            if page_text:
                text_pages.append(page_text)
        return " ".join(text_pages)
    except Exception:
        # fallback: try reading bytes and decode (rare)
        try:
            uploaded_file.seek(0)
            raw = uploaded_file.read()
            return raw.decode('utf-8', errors='ignore')
        except Exception:
            return ""


# ---------- Streamlit UI ----------
def main():
    st.title("Resume Screening App")

    uploaded_file = st.file_uploader("Upload Resume", type=['txt', 'pdf'])  # add 'docx' if supporting docx

    if uploaded_file is not None:
        # Streamlit UploadedFile supports .read(), .name, etc.
        name = uploaded_file.name.lower()

        if name.endswith('.txt'):
            uploaded_file.seek(0)
            raw_bytes = uploaded_file.read()
            try:
                resume_text = raw_bytes.decode('utf-8')
            except Exception:
                resume_text = raw_bytes.decode('latin-1', errors='ignore')

        elif name.endswith('.pdf'):
            # PyPDF2 can read the stream directly
            uploaded_file.seek(0)
            resume_text = extract_text_from_pdf(uploaded_file)
        # elif name.endswith('.docx'):
        #     # If you want to support .docx, uncomment the import docx and this block
        #     import docx
        #     uploaded_file.seek(0)
        #     with open("tmp.docx", "wb") as f:
        #         f.write(uploaded_file.read())
        #     doc = docx.Document("tmp.docx")
        #     resume_text = " ".join([p.text for p in doc.paragraphs])
        else:
            st.error("Unsupported file type")
            return

        if not resume_text.strip():
            st.error("Could not extract text from the uploaded file.")
            return

        # Clean text (pass string, not list)
        cleaned = cleanResume(resume_text)

        # Transform using tfidf and predict using classifier (correct shapes)
        X = tfidf.transform([cleaned])            # shape (1, n_features)
        try:
            pred = clf.predict(X)[0]              # get single integer label
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        # Optional: if classifier supports predict_proba, show confidence
        confidence = None
        if hasattr(clf, "predict_proba"):
            try:
                prob = clf.predict_proba(X)[0]
                confidence = prob.max()
            except Exception:
                confidence = None

        # If you have a mapping from label id -> name, put it here:
        category_mapping = {
            15: "Java Developer", 23: "Testing", 8: "DevOps Engineer",
            20: "Python Developer", 24: "Web Designing", 12: "HR",
            13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
            18: "Operations Manager", 6: "Data Science", 22: "Sales",
            16: "Mechanical Engineer", 1: "Arts", 7: "Database",
            11: "Electrical Engineering", 14: "Health and fitness",
            19: "PMO", 4: "Business Analyst", 9: "DotNet Developer",
            2: "Automation Testing", 17: "Network Security Engineer",
            21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate",
        }

        category_name = category_mapping.get(int(pred), str(pred))

        st.write("### Predicted Category")
        st.success(category_name)

        if confidence is not None:
            st.write(f"Confidence: {confidence:.2%}")

        # show cleaned preview
        with st.expander("See cleaned resume text (preview)"):
            st.write(cleaned[:2000])  # first 2000 chars

if __name__ == "__main__":
    main()

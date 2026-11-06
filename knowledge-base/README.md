# PUCIT New Campus Knowledge Base for RAG (LangChain)

A structured, grounded knowledge base designed for Retrieval-Augmented Generation (RAG) applications using **LangChain**, focusing on **Punjab University College of Information Technology (PUCIT)** — now **Faculty of Computing & Information Technology (FCIT)**, University of the Punjab, **Quaid-i-Azam Campus (New Campus), Lahore, Pakistan**.

## Directory Structure

```
knowledge_base/
├── README.md                                          # Knowledge base index & RAG ingestion guide
├── teachers/                                          # Faculty profiles across departments
│   ├── 01_cs_department_faculty.md                   # Computer Science faculty & researchers
│   ├── 02_software_engineering_faculty.md            # Software Engineering faculty & FYDP leads
│   ├── 03_information_technology_faculty.md          # Information Technology faculty & AI leads
│   ├── 04_data_science_faculty.md                    # Data Science faculty & NLP researchers
│   └── 05_faculty_directory_overview.md              # Consolidated lookup table of all teachers
├── employees/                                         # Administrative, technical & support staff
│   ├── 01_administration_and_management.md           # Principal office, Dean office, Campus management
│   ├── 02_examination_and_academic_branch.md         # Exams, transcripts, grade records & verification
│   ├── 03_it_infrastructure_and_labs.md              # Network engineers, systems admin, computer labs
│   ├── 04_library_and_learning_resources.md          # Departmental & Central PU Library resources
│   └── 05_student_affairs_and_project_office.md      # Admissions, FYDP project office, financial aid
├── students/                                          # Student academics, life & admissions
│   ├── 01_degree_programs_and_curriculum.md          # BSCS, BSSE, BSIT, BSDS, BSAI, MS, PhD programs
│   ├── 02_student_societies_and_clubs.md             # ACM, IEEE, PALS, Sports, BDS, SoftExpo, PUCON
│   ├── 03_campus_facilities_and_student_life.md       # New Campus grounds, labs, hostels, bus points
│   └── 04_admissions_merit_and_scholarships.md       # Eligibility, merit formula, Morning vs Replica
└── alumni/                                            # Alumni network, careers & achievements
    ├── 01_alumni_network_and_association.md          # Alumni association, portals, mentorship programs
    ├── 02_notable_tech_founders_and_industry_leaders.md # Tech startup founders, CEOs, incubation
    ├── 03_global_workforce_and_careers.md            # Alumni at Big Tech (Google, MS, AWS) & local firms
    └── 04_distinguished_academic_alumni.md           # Alumni in global academia, PhD researchers

```

## Content Design & RAG Optimization Principles

Each `.md` document in this repository incorporates key architectural traits for vector retrieval:

1. **Frontmatter Metadata**: Every file begins with YAML frontmatter containing `title`, `category`, `campus`, `faculty`, `department`, and search `tags`.

2. **Deterministic Heading Hierarchy**: Strict `#`, `##`, and `###` headers allow `MarkdownHeaderTextSplitter` to retain structural context across chunks.

3. **High Entity Density**: Exact real-world entity names, email addresses (`@pucit.edu.pk`), designations, research specializations, course titles, campus locations, and official reference links.

4. **Structured Tabular Matrices**: Key comparisons (e.g., Morning vs. Self-Supporting shifts, Faculty quick lookup, Career stages) are formatted as Markdown pipe tables for tabular query-answering.

## Ingesting into LangChain RAG Pipeline

Below is a complete, ready-to-run Python script using LangChain to ingest the knowledge base, split it into chunks preserving metadata, embed with vector embeddings, and query via a RAG chain:

```
import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS # or Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Define paths
KB_DIR = Path("./knowledge_base")

# 2. Load all markdown files
loader = DirectoryLoader(
    str(KB_DIR),
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
raw_documents = loader.load()
print(f"Loaded {len(raw_documents)} knowledge base documents.")

# 3. Structure-preserving splitting
headers_to_split_on = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False
)

chunked_docs = []
for doc in raw_documents:
    splits = markdown_splitter.split_text(doc.page_content)
    # Re-attach parent file metadata
    for split in splits:
        split.metadata.update(doc.metadata)
        chunked_docs.append(split)

# Further refine large sections if needed
char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)
final_chunks = char_splitter.split_documents(chunked_docs)
print(f"Created {len(final_chunks)} searchable chunks.")

# 4. Generate embeddings and initialize Vector Store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(final_chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 5. Build RAG Question-Answering Chain
llm = ChatOpenAI(model="gpt-4o", temperature=0)

system_prompt = (
    "You are an expert AI assistant answering questions about PUCIT / FCIT New Campus. "
    "Use the following pieces of retrieved context to answer the question accurately. "
    "If you don't know the answer, say you don't know.\n\n"
    "Context:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 6. Sample Query
query = "Who is the Chairman of Computer Science department and what is his research area?"
response = rag_chain.invoke({"input": query})
print("\nQuestion:", query)
print("\nAnswer:\n", response["answer"])

```
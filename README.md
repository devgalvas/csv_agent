# Vertis Data Consultant 🤖📊

A modern, interactive chatbot built with **Streamlit** that lets you upload CSV files and ask questions about your data in natural language. Powered by LLMs (via Groq API) and DuckDB for fast, flexible querying.

---

## Features

- **Upload CSV files** (supports large files)
- **Ask questions in natural language** about your data
- **Automatic SQL generation** using LLMs
- **View answers as tables or text**
- **Show raw data** in an expandable section
- **Beautiful dark blue UI** for easy reading
- **Sidebar** for quick access to upload and options

---

## Screenshots

![screenshot](image/chatbot.webp)

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/vertis-data-consultant.git
cd vertis-data-consultant/vertis_research_agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your API key

Create a `.env` file in the project root:

```
API_KEY=your_groq_api_key_here
```

Or export it in your shell:

```bash
export API_KEY=your_groq_api_key_here
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## Usage

1. **Upload a CSV file** using the sidebar.
2. **Ask a question** (e.g., "Which rows have missing values?").
3. **View the answer** as a table or text.
4. **Expand "Show raw data"** to preview your CSV.

---

## Technologies Used

- [Streamlit](https://streamlit.io/)
- [DuckDB](https://duckdb.org/)
- [Groq API](https://console.groq.com/)
- [Python](https://python.org/)

---

## Credits

Made with ❤️ by [Lucas Galvão Freitas](https://github.com/devgalvas)

---

## License

MIT License

# 🏢 Enterprise AI HR Management Suite Pro

## What This Agent Does

The **Enterprise AI HR Management Suite Pro** is a comprehensive AI-powered human resources management system that automates and enhances HR operations through intelligent document processing and multi-agent AI capabilities.

### Core Functionality:
- **Document Intelligence**: Processes HR policies, handbooks, job descriptions, and compliance documents
- **Multi-Agent System**: 4 specialized AI agents handling different HR functions
- **Real-time Analytics**: Live performance tracking and insights dashboard
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices

## 🤖 AI Agents & Features

### 1. HR Assistant Agent 👥
**What it does:**
- Answers employee questions about company policies
- Provides guidance on benefits and compensation
- Explains leave procedures and compliance requirements
- Routes urgent queries with priority handling

**Features:**
- Smart query categorization (Policy, Leave, Benefits, Compliance)
- Context-aware responses based on uploaded documents
- Priority routing for urgent requests
- ChatGPT-style conversation interface

### 2. Resume Screening Agent 📄
**What it does:**
- Automatically evaluates resumes against job descriptions
- Ranks candidates with AI-powered scoring (0-100)
- Provides detailed analysis of qualifications and fit
- Generates hiring recommendations

**Features:**
- Batch processing of multiple resumes
- Multi-factor scoring (Technical, Experience, Education, Soft Skills)
- Customizable screening thresholds
- Export results in JSON/CSV formats
- Interview scheduling integration

### 3. Interview Agent 🎤
**What it does:**
- Generates intelligent interview questions based on role and experience level
- Evaluates candidate responses in real-time
- Tracks interview performance and provides scoring
- Manages complete interview sessions

**Features:**
- Question types: Technical, Behavioral, Situational, Leadership
- Experience-level adaptation (Junior, Mid-level, Senior, Executive)
- Real-time response evaluation with detailed feedback
- Session management and progress tracking
- Performance analytics and reporting

### 4. Employee Onboarding Agent 🚀
**What it does:**
- Guides new hires through structured onboarding process
- Provides role-specific task lists and resources
- Tracks completion progress across multiple stages
- Customizes experience based on department and role

**Features:**
- 4-stage onboarding: Welcome → Documentation → Training → Setup
- Role-specific customization (Software Engineer, Data Scientist, etc.)
- Interactive task completion tracking
- Progress visualization and completion certificates
- Resource library with relevant documents

## 📊 Advanced Features

### Real-time Analytics Dashboard
- System performance metrics (Success rate, Response time, Health status)
- Interactive charts and visualizations
- User engagement tracking
- Session analytics and reporting

### Responsive Design
- Mobile-first approach with adaptive layouts
- Touch-friendly interface for tablets and phones
- Desktop optimization for full feature access
- Manual mobile/desktop view toggle

### Enterprise Integration
- Document categorization and management
- Comprehensive export capabilities
- Notification system with real-time alerts
- Multi-format reporting (JSON, CSV, PDF)

## 🛠️ Tools, APIs & Models Used

### AI Models & APIs
- **Google Gemini 2.5-Pro**: Primary language model for intelligent responses
- **Google Generative AI Embeddings**: Document vectorization and semantic search
- **LangChain Framework**: AI orchestration and document processing
- **FAISS Vector Database**: Fast similarity search and document retrieval

### Development Stack
- **Streamlit**: Web application framework and UI
- **Python**: Core programming language
- **Plotly**: Interactive data visualizations and charts
- **Pandas**: Data manipulation and analytics

### Document Processing
- **PyPDF**: PDF document extraction and processing
- **LangChain Document Loaders**: Multi-format document support (PDF, DOCX, TXT, MD)
- **Text Splitters**: Intelligent document chunking for AI processing

### UI/UX Technologies
- **Custom CSS**: Responsive design with glass morphism effects
- **CSS Animations**: Professional transitions and hover effects
- **Responsive Grid System**: Mobile-first adaptive layouts

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Google AI API key (from Google AI Studio)

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd ai-knowledgebase-agent
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Keys**
   Create a `.env` file in the root directory:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

5. **Access the Application**
   Open your browser and navigate to `http://localhost:8501`

### Deployment on Streamlit Cloud

1. **Upload Files**: Push all files to GitHub repository
2. **Add Secrets**: In Streamlit Cloud, add `GOOGLE_API_KEY` in secrets
3. **Deploy**: Connect repository and deploy automatically

## 📋 Limitations

### Current Limitations
- **Document Dependency**: Requires uploaded HR documents for optimal performance
- **API Rate Limits**: Subject to Google AI API usage limits
- **Language Support**: Currently optimized for English language documents
- **File Size Limits**: Maximum 200MB per file upload in Streamlit

### Technical Constraints
- **Vector Storage**: FAISS database is session-based (not persistent across restarts)
- **Concurrent Users**: Performance may vary with high concurrent usage
- **Mobile Features**: Some advanced features optimized for desktop use

### Data Privacy
- **Local Processing**: Documents processed locally, not stored permanently
- **API Calls**: Text sent to Google AI for processing (review Google's privacy policy)
- **Session Data**: Chat history and analytics cleared on session end

## 🎯 Use Cases

### For HR Professionals
- Automate routine policy questions and guidance
- Streamline resume screening and candidate evaluation
- Standardize interview processes with AI assistance
- Enhance new employee onboarding experience

### For Organizations
- Reduce HR workload through intelligent automation
- Improve consistency in HR processes and decisions
- Gain insights through analytics and performance tracking
- Scale HR operations efficiently

### For Employees
- Get instant answers to HR policy questions
- Access self-service guidance for benefits and procedures
- Receive personalized onboarding support
- Interact through modern, intuitive chat interface

## 🚀 Future Enhancements

### Planned Features
- Multi-language support for global organizations
- Integration with popular HRIS systems (Workday, BambooHR)
- Advanced analytics with predictive insights
- Voice interaction capabilities
- Mobile app development

### Scalability Improvements
- Persistent vector database integration
- Multi-tenant architecture for enterprise deployment
- Advanced security features and compliance certifications
- API endpoints for third-party integrations

---

**Built with ❤️ using cutting-edge AI technology for modern HR management**
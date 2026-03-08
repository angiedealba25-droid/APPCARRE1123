import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ─── Configuración de página ───
st.set_page_config(
    page_title="CareerPredict AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Estilos CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800;900&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2.5rem 2rem 2rem 2rem; border-radius: 20px; text-align: center;
        margin-bottom: 1.8rem; box-shadow: 0 12px 40px rgba(102,126,234,0.35);
        position: relative; overflow: hidden;
    }
    .hero-section::before {
        content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
        animation: shimmer 6s ease-in-out infinite;
    }
    @keyframes shimmer { 0%,100% { transform: rotate(0deg); } 50% { transform: rotate(180deg); } }
    .main-title {
        color: #ffffff !important; font-size: 5.5rem !important; font-weight: 900 !important;
        margin: 0 0 0.3rem 0 !important; padding-top: 0.3rem;
        text-shadow: 0 6px 20px rgba(0,0,0,0.3), 0 0 40px rgba(255,255,255,0.15);
        letter-spacing: -2px; position: relative; line-height: 1.1 !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    .subtitle {
        color: rgba(255,255,255,0.92) !important; font-size: 1.6rem !important; font-weight: 600 !important;
        margin-bottom: 0.8rem; position: relative;
        -webkit-text-fill-color: rgba(255,255,255,0.92) !important;
    }
    }
    .hero-desc {
        color: rgba(255,255,255,0.85) !important; font-size: 1rem; font-weight: 400;
        max-width: 700px; margin-left: auto !important; margin-right: auto !important;
        line-height: 1.7; position: relative;
        text-align: center !important; display: block;
    }
    .step-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 0.8rem 1.2rem; border-radius: 12px; font-size: 1.2rem;
        font-weight: 700; margin: 1.5rem 0 1rem 0; text-align: center;
    }
    .gold-card {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        padding: 1.8rem; border-radius: 16px; color: #1a1a1a; margin-bottom: 1rem;
        box-shadow: 0 8px 25px rgba(247,151,30,0.3); text-align: center;
    }
    .gold-card h2 { margin: 0 0 0.3rem 0; font-size: 1.5rem; }
    .gold-card .pct { font-size: 2.8rem; font-weight: 800; margin: 0; }
    .silver-card {
        background: linear-gradient(135deg, #bdc3c7 0%, #ecf0f1 100%);
        padding: 1.4rem; border-radius: 14px; color: #2c3e50; margin-bottom: 1rem;
        box-shadow: 0 6px 18px rgba(0,0,0,0.1); text-align: center;
    }
    .silver-card h3 { margin: 0 0 0.2rem 0; }
    .silver-card .pct { font-size: 2rem; font-weight: 700; margin: 0; }
    .bronze-card {
        background: linear-gradient(135deg, #b08d57 0%, #e6c88a 100%);
        padding: 1.4rem; border-radius: 14px; color: #2c2c2c; margin-bottom: 1rem;
        box-shadow: 0 6px 18px rgba(0,0,0,0.1); text-align: center;
    }
    .bronze-card h3 { margin: 0 0 0.2rem 0; }
    .bronze-card .pct { font-size: 2rem; font-weight: 700; margin: 0; }
    .reason-box {
        background: #f0f4ff; border-left: 5px solid #667eea; padding: 1rem 1.2rem;
        border-radius: 0 10px 10px 0; margin-bottom: 0.8rem; font-size: 0.95rem;
    }
    .winner-banner {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white; padding: 1.2rem; border-radius: 14px; text-align: center;
        font-size: 1.3rem; font-weight: 700; margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(17,153,142,0.3);
    }
    .emo-result {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1.5rem; border-radius: 14px; color: #1a1a1a; margin: 1rem 0;
        box-shadow: 0 8px 20px rgba(250,112,154,0.25);
    }
    .emo-result h3 { margin: 0 0 0.5rem 0; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.4rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DATOS Y MODELO
# ══════════════════════════════════════════════════════════════════════════════

MATERIAS_OPCIONES = {
    "📐 Matemáticas":      {"mat": 3, "prog": 1, "creat": 0, "lid": 0},
    "⚛️ Física":           {"mat": 2, "prog": 1, "creat": 0, "lid": 0},
    "🧬 Biología":         {"mat": 1, "prog": 0, "creat": 1, "lid": 1},
    "🧪 Química":          {"mat": 2, "prog": 0, "creat": 1, "lid": 0},
    "📜 Historia":         {"mat": 0, "prog": 0, "creat": 2, "lid": 2},
    "💻 Programación":     {"mat": 1, "prog": 3, "creat": 1, "lid": 0},
    "🎨 Arte y Diseño":    {"mat": 0, "prog": 0, "creat": 3, "lid": 1},
    "💼 Economía":         {"mat": 2, "prog": 1, "creat": 0, "lid": 2},
    "📖 Literatura":       {"mat": 0, "prog": 0, "creat": 3, "lid": 1},
    "🌍 Idiomas":          {"mat": 0, "prog": 0, "creat": 2, "lid": 2},
    "🔬 Laboratorio":      {"mat": 1, "prog": 1, "creat": 1, "lid": 0},
    "🏋️ Educación Física": {"mat": 0, "prog": 0, "creat": 1, "lid": 3},
    "🎵 Música":           {"mat": 1, "prog": 0, "creat": 3, "lid": 0},
    "🗣️ Debate / Oratoria":{"mat": 0, "prog": 0, "creat": 2, "lid": 3},
    "🤖 Robótica":         {"mat": 2, "prog": 3, "creat": 2, "lid": 0},
    "📊 Estadística":      {"mat": 3, "prog": 2, "creat": 0, "lid": 0},
}

HABILIDADES_OPCIONES = {
    "🧠 Resolver problemas lógicos":       {"mat": 3, "prog": 2, "creat": 0, "lid": 0},
    "💡 Inventar o crear cosas nuevas":     {"mat": 0, "prog": 1, "creat": 3, "lid": 1},
    "👥 Liderar equipos de trabajo":        {"mat": 0, "prog": 0, "creat": 1, "lid": 3},
    "💻 Programar o usar computadoras":     {"mat": 1, "prog": 3, "creat": 1, "lid": 0},
    "🗣️ Comunicar ideas y persuadir":       {"mat": 0, "prog": 0, "creat": 2, "lid": 3},
    "🔬 Experimentar e investigar":         {"mat": 2, "prog": 1, "creat": 2, "lid": 0},
    "📈 Analizar datos y números":          {"mat": 3, "prog": 2, "creat": 0, "lid": 0},
    "🎨 Dibujar, diseñar o expresarme":     {"mat": 0, "prog": 0, "creat": 3, "lid": 1},
    "🤝 Ayudar y cuidar a las personas":    {"mat": 0, "prog": 0, "creat": 1, "lid": 2},
    "🏗️ Construir o armar cosas":           {"mat": 2, "prog": 1, "creat": 2, "lid": 0},
    "📋 Organizar y planificar proyectos":  {"mat": 1, "prog": 0, "creat": 1, "lid": 3},
    "🌐 Trabajar con redes sociales":       {"mat": 0, "prog": 2, "creat": 3, "lid": 1},
}

AREA_OPCIONES = {
    "🖥️ Tecnología e Informática":  "Tecnología",
    "🏥 Salud y Ciencias de la Vida": "Salud",
    "📊 Negocios y Emprendimiento":  "Negocios",
    "⚙️ Ingeniería y Construcción":  "Ingeniería",
}

# Preguntas del Test Emocional
TEST_EMOCIONAL = [
    {
        "pregunta": "Cuando tienes un problema difícil, ¿qué haces primero?",
        "opciones": {
            "🔍 Lo analizo paso a paso con lógica":                {"analitico": 3, "creativo": 0, "social": 0, "practico": 1},
            "💡 Busco una solución creativa o diferente":          {"analitico": 0, "creativo": 3, "social": 0, "practico": 1},
            "🗣️ Pido opinión a amigos o compañeros":               {"analitico": 0, "creativo": 0, "social": 3, "practico": 1},
            "🔧 Me pongo a intentar soluciones hasta que funcione":{"analitico": 1, "creativo": 1, "social": 0, "practico": 3},
        },
    },
    {
        "pregunta": "Un sábado libre ideal para ti sería...",
        "opciones": {
            "📚 Aprender algo nuevo (curso, documental, libro)":  {"analitico": 3, "creativo": 1, "social": 0, "practico": 0},
            "🎨 Crear algo: pintar, escribir, diseñar, cocinar":  {"analitico": 0, "creativo": 3, "social": 1, "practico": 0},
            "🎉 Reunirme con amigos o ir a un evento social":     {"analitico": 0, "creativo": 0, "social": 3, "practico": 1},
            "🏕️ Hacer algo al aire libre o una actividad física": {"analitico": 0, "creativo": 1, "social": 1, "practico": 3},
        },
    },
    {
        "pregunta": "En un trabajo en equipo, tú usualmente...",
        "opciones": {
            "📊 Organizo los datos y la información":             {"analitico": 3, "creativo": 0, "social": 1, "practico": 0},
            "🎯 Propongo las ideas originales":                   {"analitico": 0, "creativo": 3, "social": 1, "practico": 0},
            "👑 Coordino al equipo y distribuyo tareas":          {"analitico": 0, "creativo": 0, "social": 3, "practico": 1},
            "⚡ Ejecuto las tareas lo más rápido posible":        {"analitico": 1, "creativo": 0, "social": 0, "practico": 3},
        },
    },
    {
        "pregunta": "¿Qué tipo de película/serie te atrapa más?",
        "opciones": {
            "🔬 Ciencia ficción o misterio":                      {"analitico": 3, "creativo": 1, "social": 0, "practico": 0},
            "🌈 Fantasía, animación o mundos creativos":          {"analitico": 0, "creativo": 3, "social": 1, "practico": 0},
            "❤️ Drama, romance o historias de personas reales":   {"analitico": 0, "creativo": 1, "social": 3, "practico": 0},
            "💥 Acción, aventura o deportes":                     {"analitico": 0, "creativo": 0, "social": 1, "practico": 3},
        },
    },
    {
        "pregunta": "Si pudieras tener un superpoder, ¿cuál elegirías?",
        "opciones": {
            "🧠 Super inteligencia":                              {"analitico": 3, "creativo": 1, "social": 0, "practico": 0},
            "✨ Crear cualquier cosa de la nada":                  {"analitico": 0, "creativo": 3, "social": 0, "practico": 1},
            "💞 Leer las emociones de los demás":                 {"analitico": 0, "creativo": 0, "social": 3, "practico": 1},
            "💪 Super fuerza o velocidad":                        {"analitico": 0, "creativo": 0, "social": 1, "practico": 3},
        },
    },
    {
        "pregunta": "¿Qué frase te identifica más?",
        "opciones": {
            "📏 'Los datos no mienten'":                          {"analitico": 3, "creativo": 0, "social": 0, "practico": 1},
            "🌟 'La imaginación es más importante que el saber'": {"analitico": 0, "creativo": 3, "social": 1, "practico": 0},
            "🤗 'Juntos llegamos más lejos'":                     {"analitico": 0, "creativo": 0, "social": 3, "practico": 1},
            "🚀 'Hecho es mejor que perfecto'":                   {"analitico": 1, "creativo": 0, "social": 0, "practico": 3},
        },
    },
]

PERFIL_EMOCIONAL_DESC = {
    "analitico": {
        "emoji": "🧠", "nombre": "Analítico",
        "desc": "Eres una persona lógica, metódica y orientada a los datos. Disfrutas resolver problemas complejos y encontrar patrones. Las carreras que requieren pensamiento crítico y precisión son ideales para ti.",
        "carreras": ["Ciencia de Datos", "Ingeniería en Sistemas", "Economía", "Ingeniería Mecatrónica", "Biotecnología"],
    },
    "creativo": {
        "emoji": "🎨", "nombre": "Creativo",
        "desc": "Tu mente siempre está generando ideas nuevas. Te expresas a través del diseño, la innovación y el arte. Buscas carreras donde puedas imaginar, inventar y transformar el mundo con originalidad.",
        "carreras": ["Diseño Gráfico", "Arquitectura", "Marketing Digital", "Ingeniería Mecatrónica", "Psicología"],
    },
    "social": {
        "emoji": "💞", "nombre": "Social",
        "desc": "Las personas son tu motor. Te encanta comunicar, ayudar, liderar y conectar con los demás. Las carreras que involucran trabajo en equipo, servicio y relaciones humanas son tu camino.",
        "carreras": ["Medicina", "Psicología", "Derecho", "Administración de Empresas", "Enfermería"],
    },
    "practico": {
        "emoji": "🔧", "nombre": "Práctico",
        "desc": "Prefieres la acción sobre la teoría. Te gusta construir, reparar, ejecutar y ver resultados tangibles. Las carreras técnicas y de campo donde puedas aplicar conocimientos de inmediato son perfectas.",
        "carreras": ["Ingeniería Civil", "Ingeniería Industrial", "Ingeniería Mecatrónica", "Enfermería", "Biotecnología"],
    },
}


@st.cache_data
def generar_dataset(n=600):
    np.random.seed(42)
    carreras = list({
        "Ingeniería en Sistemas", "Medicina", "Administración de Empresas",
        "Ingeniería Civil", "Diseño Gráfico", "Psicología", "Ciencia de Datos",
        "Derecho", "Arquitectura", "Ingeniería Industrial", "Biotecnología",
        "Marketing Digital", "Economía", "Enfermería", "Ingeniería Mecatrónica",
    })
    carreras.sort()

    perfiles = {
        # Salarios basados en: virtual.cuc.edu.co (2026) y poli.edu.co (2026) — COP mensuales
        "Administración de Empresas": {"mat": (5, 8),  "prog": (2, 5),  "creat": (5, 8), "lid": (7, 10), "sal": (2500000, 6000000),    "area": "Negocios",   "demanda": 94},
        "Arquitectura":               {"mat": (6, 9),  "prog": (2, 5),  "creat": (8, 10),"lid": (4, 7),  "sal": (2300000, 5500000),    "area": "Ingeniería",  "demanda": 65},
        "Biotecnología":              {"mat": (7, 10), "prog": (3, 6),  "creat": (5, 8), "lid": (3, 6),  "sal": (2300000, 5500000),    "area": "Salud",       "demanda": 85},
        "Ciencia de Datos":           {"mat": (8, 10), "prog": (8, 10), "creat": (4, 7), "lid": (3, 7),  "sal": (3500000, 12000000),   "area": "Tecnología",  "demanda": 98},
        "Derecho":                    {"mat": (3, 6),  "prog": (1, 3),  "creat": (4, 7), "lid": (7, 10), "sal": (2800000, 9000000),    "area": "Negocios",    "demanda": 82},
        "Diseño Gráfico":             {"mat": (2, 5),  "prog": (3, 6),  "creat": (8, 10),"lid": (3, 6),  "sal": (2000000, 5000000),    "area": "Tecnología",  "demanda": 70},
        "Economía":                   {"mat": (7, 10), "prog": (3, 6),  "creat": (3, 6), "lid": (5, 8),  "sal": (2500000, 6000000),    "area": "Negocios",    "demanda": 74},
        "Enfermería":                 {"mat": (4, 7),  "prog": (1, 3),  "creat": (3, 6), "lid": (5, 8),  "sal": (2000000, 3800000),    "area": "Salud",       "demanda": 92},
        "Ingeniería Civil":           {"mat": (8, 10), "prog": (3, 6),  "creat": (4, 7), "lid": (4, 7),  "sal": (2800000, 7000000),    "area": "Ingeniería",  "demanda": 75},
        "Ingeniería Industrial":      {"mat": (7, 10), "prog": (4, 7),  "creat": (4, 7), "lid": (6, 9),  "sal": (3000000, 10000000),   "area": "Ingeniería",  "demanda": 88},
        "Ingeniería Mecatrónica":     {"mat": (8, 10), "prog": (7, 10), "creat": (5, 8), "lid": (3, 7),  "sal": (2800000, 8000000),    "area": "Ingeniería",  "demanda": 87},
        "Ingeniería en Sistemas":     {"mat": (7, 10), "prog": (8, 10), "creat": (4, 7), "lid": (3, 7),  "sal": (3500000, 15000000),   "area": "Tecnología",  "demanda": 84},
        "Marketing Digital":          {"mat": (3, 6),  "prog": (4, 7),  "creat": (7, 10),"lid": (6, 9),  "sal": (2500000, 10000000),   "area": "Negocios",    "demanda": 85},
        "Medicina":                   {"mat": (6, 9),  "prog": (1, 4),  "creat": (3, 6), "lid": (5, 9),  "sal": (3000000, 12000000),   "area": "Salud",       "demanda": 94},
        "Psicología":                 {"mat": (3, 6),  "prog": (1, 3),  "creat": (5, 8), "lid": (6, 9),  "sal": (2300000, 5000000),    "area": "Salud",       "demanda": 85},
    }

    area_map = {"Tecnología": 0, "Salud": 1, "Negocios": 2, "Ingeniería": 3}
    filas = []
    for _ in range(n):
        carrera = np.random.choice(carreras)
        p = perfiles[carrera]
        filas.append({
            "matematicas":      np.random.randint(p["mat"][0], p["mat"][1] + 1),
            "programacion":     np.random.randint(p["prog"][0], p["prog"][1] + 1),
            "creatividad":      np.random.randint(p["creat"][0], p["creat"][1] + 1),
            "liderazgo":        np.random.randint(p["lid"][0], p["lid"][1] + 1),
            "salario_esperado": np.random.randint(p["sal"][0], p["sal"][1] + 1),
            "area_interes":     area_map[p["area"]],
            "carrera":          carrera,
            "salario_promedio": int(np.mean(p["sal"])),
            "demanda_laboral":  p["demanda"],
        })
    return pd.DataFrame(filas), area_map, perfiles


@st.cache_resource
def entrenar_modelo():
    df, area_map, perfiles = generar_dataset()
    le = LabelEncoder()
    df["carrera_encoded"] = le.fit_transform(df["carrera"])
    features = ["matematicas", "programacion", "creatividad", "liderazgo",
                "salario_esperado", "area_interes"]
    X = df[features]
    y = df["carrera_encoded"]
    modelo = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
    modelo.fit(X, y)
    return modelo, le, df, area_map, perfiles


modelo, le, df, area_map, perfiles = entrenar_modelo()


def fmt_cop(valor):
    """Formatea un valor numérico como salario colombiano: $3.500.000"""
    return f"${valor:,.0f}".replace(",", ".")


def calcular_habilidades(materias_sel, habilidades_sel):
    scores = {"mat": 0, "prog": 0, "creat": 0, "lid": 0}
    for m in materias_sel:
        for k, v in MATERIAS_OPCIONES[m].items():
            scores[k] += v
    for h in habilidades_sel:
        for k, v in HABILIDADES_OPCIONES[h].items():
            scores[k] += v
    maximo = max(max(scores.values()), 1)
    return {k: min(int(round(v / maximo * 10)), 10) for k, v in scores.items()}


def generar_razones(carrera, scores, salario_esp, area_sel, perfil):
    razones = []
    mat_ok = scores["mat"] >= perfil["mat"][0]
    prog_ok = scores["prog"] >= perfil["prog"][0]
    creat_ok = scores["creat"] >= perfil["creat"][0]
    lid_ok = scores["lid"] >= perfil["lid"][0]

    if mat_ok and prog_ok:
        razones.append(f"Tu nivel de Matemáticas ({scores['mat']}/10) y Programación ({scores['prog']}/10) encajan perfectamente con lo que exige **{carrera}**.")
    elif mat_ok:
        razones.append(f"Tu fortaleza en Matemáticas ({scores['mat']}/10) es clave para tener éxito en **{carrera}**.")
    elif prog_ok:
        razones.append(f"Tu habilidad en Programación ({scores['prog']}/10) coincide con el perfil ideal de **{carrera}**.")

    if creat_ok:
        razones.append(f"Tu Creatividad ({scores['creat']}/10) te permite destacar en las áreas innovadoras de **{carrera}**.")
    if lid_ok:
        razones.append(f"Tu capacidad de Liderazgo ({scores['lid']}/10) es valorada en el campo de **{carrera}**.")

    sal_prom = int(np.mean(perfil["sal"]))
    if salario_esp <= perfil["sal"][1]:
        razones.append(f"Tu expectativa salarial ({fmt_cop(salario_esp)}) es realista: el rango de **{carrera}** va de {fmt_cop(perfil['sal'][0])} a {fmt_cop(perfil['sal'][1])} COP.")

    if AREA_OPCIONES.get(area_sel) == perfil["area"]:
        razones.append(f"Tu área de interés (**{AREA_OPCIONES[area_sel]}**) coincide directamente con el campo de esta carrera.")

    if perfil["demanda"] >= 85:
        razones.append(f"**{carrera}** tiene una demanda laboral del {perfil['demanda']}% — ¡excelentes oportunidades de empleo!")
    elif perfil["demanda"] >= 70:
        razones.append(f"**{carrera}** cuenta con buena demanda laboral ({perfil['demanda']}%).")

    if not razones:
        razones.append(f"Tu combinación general de habilidades muestra afinidad con el perfil de **{carrera}**.")

    return razones


# ══════════════════════════════════════════════════════════════════════════════
#  INTERFAZ
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-section">
    <h1 class="main-title">🎓 CareerPredict IA</h1>
    <h2 class="subtitle">Responde las preguntas y descubre la carrera ideal para ti con IA 🚀</h2>
    <p class="hero-desc">
        Esta app fue creada para ayudar a estudiantes a descubrir qué carrera universitaria
        se adapta mejor a sus habilidades, intereses y personalidad. Usando inteligencia artificial
        y un test emocional, te orientamos hacia tu futuro profesional ideal en Colombia.
    </p>
</div>
""", unsafe_allow_html=True)

# ── PASO 1: Materias ──
st.markdown('<div class="step-title">📚 PASO 1 — ¿Qué materias te gustan más?</div>', unsafe_allow_html=True)
st.caption("Selecciona todas las que disfrutes. No hay respuestas correctas ni incorrectas.")

cols_mat = st.columns(4)
materias_sel = []
for i, (materia, _) in enumerate(MATERIAS_OPCIONES.items()):
    with cols_mat[i % 4]:
        if st.checkbox(materia, key=f"mat_{i}"):
            materias_sel.append(materia)

# ── PASO 2: Habilidades ──
st.markdown('<div class="step-title">💪 PASO 2 — ¿En qué eres bueno/a?</div>', unsafe_allow_html=True)
st.caption("Elige las habilidades que mejor te describen.")

cols_hab = st.columns(3)
habilidades_sel = []
for i, (hab, _) in enumerate(HABILIDADES_OPCIONES.items()):
    with cols_hab[i % 3]:
        if st.checkbox(hab, key=f"hab_{i}"):
            habilidades_sel.append(hab)

# ── PASO 3: Salario y Área ──
st.markdown('<div class="step-title">🎯 PASO 3 — Expectativas y Área de interés</div>', unsafe_allow_html=True)

col_s, col_a = st.columns(2)
with col_s:
    salario_esperado = st.select_slider(
        "💰 ¿Cuánto te gustaría ganar al mes? (COP)",
        options=list(range(1500000, 15500000, 500000)),
        value=3000000,
        format_func=lambda x: f"${x:,}",
    )
with col_a:
    area_interes = st.radio(
        "🌐 ¿Qué área te llama más la atención?",
        list(AREA_OPCIONES.keys()),
        horizontal=True,
    )

# ── BOTÓN PREDECIR ──
st.markdown("")
col_btn = st.columns([1, 2, 1])
with col_btn[1]:
    predecir = st.button("🚀 ¡Predecir mi carrera ideal!", width='stretch', type="primary")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
#  RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════

# Guardar resultados en session_state para que persistan entre interacciones
if predecir:
    if len(materias_sel) == 0 and len(habilidades_sel) == 0:
        st.warning("⚠️ Selecciona al menos una materia o una habilidad para obtener tu predicción.")
        st.stop()

    scores = calcular_habilidades(materias_sel, habilidades_sel)
    area_code = area_map[AREA_OPCIONES[area_interes]]

    entrada = np.array([[scores["mat"], scores["prog"], scores["creat"],
                          scores["lid"], salario_esperado, area_code]])
    probabilidades = modelo.predict_proba(entrada)[0]
    top3_idx = probabilidades.argsort()[-3:][::-1]
    top3_carreras = le.inverse_transform(top3_idx)
    top3_probs = probabilidades[top3_idx]

    st.session_state["resultado"] = {
        "scores": scores,
        "top3_carreras": top3_carreras,
        "top3_probs": top3_probs,
        "salario_esperado": salario_esperado,
        "area_interes": area_interes,
        "materias_sel": materias_sel,
        "habilidades_sel": habilidades_sel,
    }

if "resultado" in st.session_state:
    res = st.session_state["resultado"]
    scores = res["scores"]
    top3_carreras = res["top3_carreras"]
    top3_probs = res["top3_probs"]
    salario_esperado = res["salario_esperado"]
    area_interes = res["area_interes"]
    materias_sel = res["materias_sel"]
    habilidades_sel = res["habilidades_sel"]

    mejor = top3_carreras[0]
    info_mejor = perfiles[mejor]

    st.markdown(f"""
    <div class="winner-banner">
        🏆 ¡Tu carrera más opcionada es: {mejor}!  —  {top3_probs[0]*100:.1f}% de compatibilidad
    </div>
    """, unsafe_allow_html=True)

    # ── Tarjetas Top 3 ──
    st.markdown("### 🎯 Tus 3 carreras más compatibles")
    card_classes = ["gold-card", "silver-card", "bronze-card"]
    medallas = ["🥇", "🥈", "🥉"]
    tag_pct = ["pct", "pct", "pct"]

    c1, c2, c3 = st.columns(3)
    for i, col in enumerate([c1, c2, c3]):
        carrera_n = top3_carreras[i]
        prob = top3_probs[i]
        info = perfiles[carrera_n]
        heading = "h2" if i == 0 else "h3"
        with col:
            st.markdown(f"""
            <div class="{card_classes[i]}">
                <{heading}>{medallas[i]} {carrera_n}</{heading}>
                <p class="pct">{prob*100:.1f}%</p>
                <p style="margin:0.2rem 0 0 0;">Área: {info['area']} · Demanda: {info['demanda']}%</p>
                <p style="margin:0;">Salario: {fmt_cop(info['sal'][0])} – {fmt_cop(info['sal'][1])} COP</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Gráfico de barras comparativo ──
    st.markdown("### 📊 Comparación de las 3 carreras")
    colores = ["#f7971e", "#95a5a6", "#b08d57"]

    df_comp = pd.DataFrame({
        "Carrera": list(top3_carreras),
        "Compatibilidad (%)": [round(p * 100, 1) for p in top3_probs],
        "Salario Promedio (COP)": [int(np.mean(perfiles[c]["sal"])) for c in top3_carreras],
        "Demanda Laboral (%)": [perfiles[c]["demanda"] for c in top3_carreras],
    })

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name="Compatibilidad (%)", x=df_comp["Carrera"], y=df_comp["Compatibilidad (%)"],
        marker_color=colores, text=df_comp["Compatibilidad (%)"].apply(lambda x: f"{x}%"),
        textposition="outside",
    ))
    fig_comp.update_layout(
        yaxis_title="Compatibilidad (%)", yaxis_range=[0, max(top3_probs) * 100 + 15],
        template="plotly_white", height=420,
        title={
            "text": f"🏆 {mejor} es la carrera #1 para tu perfil",
            "x": 0.5, "xanchor": "center",
            "font": {"size": 18, "color": "#1a1a1a"},
        },
    )
    st.plotly_chart(fig_comp, width='stretch')

    # Gráfico comparativo salario + demanda
    fig_multi = go.Figure()
    fig_multi.add_trace(go.Bar(
        name="Salario Promedio (miles COP)",
        x=df_comp["Carrera"],
        y=df_comp["Salario Promedio (COP)"].apply(lambda x: x / 1000),
        marker_color=["#667eea", "#764ba2", "#f093fb"],
        text=df_comp["Salario Promedio (COP)"].apply(lambda x: fmt_cop(x)),
        textposition="outside",
    ))
    fig_multi.add_trace(go.Bar(
        name="Demanda Laboral (%)",
        x=df_comp["Carrera"],
        y=df_comp["Demanda Laboral (%)"],
        marker_color=["#11998e", "#38ef7d", "#56ccf2"],
        text=df_comp["Demanda Laboral (%)"].apply(lambda x: f"{x}%"),
        textposition="outside",
    ))
    fig_multi.update_layout(
        barmode="group", template="plotly_white", height=420,
        yaxis_title="Valor", title={"text": "Salario vs Demanda laboral", "x": 0.5, "xanchor": "center"},
    )
    st.plotly_chart(fig_multi, width='stretch')

    # ── Razones detalladas por cada carrera ──
    st.markdown("### 💡 ¿Por qué estas carreras son para ti?")

    for i, carrera_n in enumerate(top3_carreras):
        perfil_c = perfiles[carrera_n]
        razones = generar_razones(carrera_n, scores, salario_esperado, area_interes, perfil_c)
        with st.expander(f"{medallas[i]} **{carrera_n}** — {top3_probs[i]*100:.1f}% compatibilidad", expanded=(i == 0)):
            for r in razones:
                st.markdown(f'<div class="reason-box">✅ {r}</div>', unsafe_allow_html=True)

    # ── Radar de habilidades ──
    st.markdown("### 🕸️ Tu perfil de habilidades calculado")
    cats = ["Matemáticas", "Programación", "Creatividad", "Liderazgo"]
    vals = [scores["mat"], scores["prog"], scores["creat"], scores["lid"]]
    fig_radar = px.line_polar(
        r=vals + [vals[0]], theta=cats + [cats[0]], line_close=True,
    )
    fig_radar.update_traces(fill="toself", fillcolor="rgba(102,126,234,0.25)", line_color="#667eea")
    fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0, 10])), height=420)
    st.plotly_chart(fig_radar, width='stretch')

    # ══════════════════════════════════════════════════════════════════════════
    #  TEST EMOCIONAL
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown('<div class="step-title">🧩 TEST EMOCIONAL — Descubre tu personalidad vocacional</div>', unsafe_allow_html=True)
    st.caption("Responde estas 6 preguntas y tu resultado emocional aparecerá automáticamente. 🎯")

    respuestas_emo = {}
    for idx, pregunta_data in enumerate(TEST_EMOCIONAL):
        st.markdown(f"**{idx + 1}. {pregunta_data['pregunta']}**")
        resp = st.radio(
            f"Elige una opción:", list(pregunta_data["opciones"].keys()),
            key=f"emo_{idx}", label_visibility="collapsed",
        )
        respuestas_emo[idx] = resp

    # Calcular resultados automáticamente (sin botón)
    emo_scores = {"analitico": 0, "creativo": 0, "social": 0, "practico": 0}
    for idx, resp in respuestas_emo.items():
        puntajes = TEST_EMOCIONAL[idx]["opciones"][resp]
        for k, v in puntajes.items():
            emo_scores[k] += v

    perfil_dominante = max(emo_scores, key=emo_scores.get)
    perfil_info = PERFIL_EMOCIONAL_DESC[perfil_dominante]
    total_emo = max(sum(emo_scores.values()), 1)

    st.markdown("---")
    st.markdown(f"""
    <div class="emo-result">
        <h3>{perfil_info['emoji']} Tu perfil emocional dominante: {perfil_info['nombre']}</h3>
        <p style="font-size:1rem;">{perfil_info['desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Gráfico de perfil emocional
    emo_df = pd.DataFrame({
        "Dimensión": ["🧠 Analítico", "🎨 Creativo", "💞 Social", "🔧 Práctico"],
        "Puntaje": [emo_scores["analitico"], emo_scores["creativo"],
                    emo_scores["social"], emo_scores["practico"]],
    })
    emo_df["Porcentaje"] = (emo_df["Puntaje"] / total_emo * 100).round(1)

    fig_emo = px.bar(
        emo_df, x="Dimensión", y="Porcentaje",
        color="Dimensión",
        color_discrete_sequence=["#667eea", "#f093fb", "#fa709a", "#11998e"],
        text=emo_df["Porcentaje"].apply(lambda x: f"{x}%"),
    )
    fig_emo.update_traces(textposition="outside")
    fig_emo.update_layout(
        showlegend=False, template="plotly_white", height=380,
        yaxis_range=[0, max(emo_df["Porcentaje"]) + 15],
        title={"text": "Tu distribución emocional", "x": 0.5},
    )
    st.plotly_chart(fig_emo, width='stretch')

    # Carreras recomendadas por perfil emocional
    st.markdown(f"### 🎯 Carreras afines a tu perfil **{perfil_info['nombre']}**")
    carreras_emo = perfil_info["carreras"]

    # Cruzar con la predicción del modelo
    coincidencias = [c for c in top3_carreras if c in carreras_emo]

    if coincidencias:
        st.success(f"🎉 ¡**{'**, **'.join(coincidencias)}** {'aparece' if len(coincidencias)==1 else 'aparecen'} tanto en tu predicción de IA como en tu perfil emocional! Esto refuerza que {'es una' if len(coincidencias)==1 else 'son'} excelente{'s' if len(coincidencias)>1 else ''} opción{'es' if len(coincidencias)>1 else ''} para ti.")
    else:
        st.info("Tu perfil emocional sugiere carreras complementarias a tu predicción de IA. ¡Explóralas también!")

    for c in carreras_emo:
        if c in perfiles:
            p = perfiles[c]
            emoji_match = "⭐" if c in list(top3_carreras) else "📌"
            st.markdown(f"""
            <div class="reason-box">
                {emoji_match} <b>{c}</b> — Área: {p['area']} · Salario: {fmt_cop(int(np.mean(p['sal'])))} COP · Demanda: {p['demanda']}%
                {"<br>🔥 <i>¡También está en tu Top 3 de IA!</i>" if c in list(top3_carreras) else ""}
            </div>
            """, unsafe_allow_html=True)

    # Conclusión final
    st.markdown("---")
    st.markdown("### 🎓 Conclusión final")
    st.markdown(f"""
    Combinando los resultados del **modelo de IA** y tu **test emocional**:

    - 🏆 **Carrera más recomendada:** **{mejor}** ({top3_probs[0]*100:.1f}% compatibilidad)
    - {perfil_info['emoji']} **Perfil emocional:** {perfil_info['nombre']}
    - 📊 **Materias seleccionadas:** {len(materias_sel)} | **Habilidades elegidas:** {len(habilidades_sel)}

    {"🌟 **Tu perfil emocional y la IA coinciden**, lo cual es una señal muy fuerte de que vas por buen camino." if mejor in carreras_emo else f"💡 La IA sugiere **{mejor}** mientras tu perfil emocional apunta a carreras del área **{perfil_info['nombre']}**. Ambas perspectivas son valiosas — considera explorar carreras que combinen ambos enfoques."}

    > *Recuerda: esta herramienta es una guía. La mejor decisión la tomas tú informándote, explorando y siguiendo tu pasión.* ❤️
    """)

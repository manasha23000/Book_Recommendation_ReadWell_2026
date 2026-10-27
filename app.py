import html

import joblib
import pandas as pd
import re
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="ReadWell Staff Console", page_icon="📚", layout="wide")

# ---------------------------------------------------------------
# Data (loaded once, then kept in memory)
# ---------------------------------------------------------------
GENRE_WORDS = ["romance", "fantasy", "mystery"]
TOP_TAGS_FOR_GENRE = 6  # only the most common tags decide a book's genre


def genre_list(tag_string):
    tags = [t.strip().lower() for t in str(tag_string).split(",")][:TOP_TAGS_FOR_GENRE]
    return [g for g in GENRE_WORDS if any(g in t for t in tags)]


@st.cache_resource
def load_data():
    books = pd.read_pickle("books_app.pkl")
    books["genres"] = books["clean_tags_str"].map(genre_list)
    matrix = joblib.load("tfidf_app.joblib")
    sent = pd.read_csv("book_sentiment.csv", dtype={"book_id": str}).set_index("book_id")
    return books, matrix, sent


books_df, tfidf_matrix, sent_lookup = load_data()

# label shown to staff, text colour, background colour, small symbol in front of the label
TIERS = {
    "Strong reception": ("Readers loved this one", "#B3123F", "#FDE7EE", "♥"),
    "Typical reception": ("Readers liked this one", "#23806F", "#E4F4F0", "●"),
    "Weaker reception": ("Mixed reactions", "#A9650F", "#FFF1D9", "●"),
    "Not enough reviews": ("Not enough reviews yet", "#5F6675", "#EEF0F4", "●"),
    "No reviews": ("No reviews in the data", "#5F6675", "#EEF0F4", "●"),
}

TAG_STYLES = [("#6C4BD8", "#EFEAFE"), ("#D6336C", "#FDE8F0"), ("#1C7ED6", "#E5F1FD")]

# ---------------------------------------------------------------
# Style
# ---------------------------------------------------------------
st.markdown(
    """
<style>
.stApp {background:#E6E9F2;}
[data-testid="stExpander"] {background:#FFFFFF; border-radius:12px;}
/* white result cards: a hidden marker inside each card lets the style find it */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .card-marker),
[data-testid="stVerticalBlock"]:has(> .element-container .card-marker) {background:#FFFFFF; border-color:#D3D8E6 !important;}
[data-testid="stElementContainer"]:has(.card-marker), .element-container:has(.card-marker) {display:none;}
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stToolbarActions"],
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"], [data-testid="stDecoration"],
.stAppToolbar, #MainMenu, footer {display:none !important;}
.block-container {max-width: 1150px; padding-top: 2rem;}
.topbar {display:flex; align-items:center; justify-content:space-between; gap:20px;
         background:linear-gradient(120deg,#F8B4C7 0%,#FBC6D5 55%,#E0C9FA 100%); border:1px solid #F0A0B8; border-radius:18px; padding:30px 30px; margin:6px 0 18px 0; overflow:hidden;}
.brand {display:flex; align-items:center; gap:12px; min-width:0;}
.brand-logo {font-size:48px;}
.brand-name {font-size:36px; font-weight:800; color:#1B2340; line-height:1.1;}
.brand-sub {font-size:16px; color:#5F6675; line-height:1.1;}
.tagline {font-size:16px; color:#4A5170; margin-top:6px;}
.picks-box {background:#F3F0FF; border:1px solid #E2DBFF; border-radius:12px; padding:10px 16px; min-width:230px; font-size:13px; color:#3B3566;}
.topbar {flex-wrap:wrap;}
@media (max-width: 700px) {
  .block-container {padding-left:1rem; padding-right:1rem;}
  .topbar {flex-direction:column; align-items:stretch; gap:12px; padding:14px 16px;}
  .brand-name {font-size:26px;}
  .brand-logo {font-size:38px;}
  .tagline {font-size:14px;}
  .picks-box {min-width:0; width:100%; box-sizing:border-box;}
  .hero {padding:12px 16px;}
  .hero h2 {font-size:18px;}
  .cover {width:84px; height:124px;}
}
.hero {background:linear-gradient(120deg,#FFC9D6 0%,#FFDDE6 55%,#EBDDFF 100%); border-radius:14px; padding:14px 22px; margin-bottom:6px;}
.hero h2 {margin:0 0 2px 0; font-size:21px; color:#1B2340;}
.hero p {margin:0; color:#4A5170; font-size:14px;}
.section-title {font-size:24px; font-weight:800; color:#1B2340; margin:22px 0 2px 0;}
.section-sub {font-size:14px; color:#5F6675; margin-bottom:12px;}
.rank {display:inline-flex; align-items:center; justify-content:center; width:28px; height:28px; border-radius:50%;
       background:#EEF0F8; color:#1B2340; font-weight:700; font-size:14px; margin-bottom:6px;}
.b-title {font-size:19px; font-weight:700; color:#1B2340; line-height:1.25;}
.b-author {font-size:14px; color:#5F6675; margin-bottom:6px;}
.b-desc {font-size:13.5px; color:#3D4460; margin-top:6px; line-height:1.45;}
.desc summary {list-style:none; cursor:pointer; outline:none;}
.desc summary::-webkit-details-marker {display:none;}
.desc .d-full {display:none;}
.desc[open] .d-short {display:none;}
.desc[open] .d-full {display:inline;}
.see {color:#6C4BD8; font-weight:700; margin-left:4px; white-space:nowrap;}
.see:hover {text-decoration:underline;}
.tag {display:inline-block; font-size:12px; font-weight:600; padding:3px 11px; border-radius:999px; margin:0 6px 6px 0;}
.cover {width:96px; height:142px; object-fit:cover; border-radius:6px; box-shadow:0 2px 8px rgba(0,0,0,0.22);}
.cover-ph {display:flex; align-items:center; justify-content:center; text-align:center; padding:8px; box-sizing:border-box;
           background:linear-gradient(160deg,#6C4BD8,#D6336C); color:#FFFFFF; font-size:12px; font-weight:700; line-height:1.3;}
.sent {border-radius:12px; padding:12px 14px; margin-bottom:8px;}
.sent-title {font-size:15px; font-weight:800;}
.sent-sub {font-size:12.5px; margin-top:2px; opacity:0.85;}
.note {background:#EAF1FB; border-radius:14px; padding:16px 20px; color:#34405F; font-size:14px; margin-top:18px; line-height:1.5;}
.note b {color:#1B2340;}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def esc(value):
    return html.escape(str(value)) if value is not None else ""


def tags_html(tag_string, n=3):
    if not isinstance(tag_string, str) or not tag_string.strip():
        return ""
    tags = [t.strip() for t in tag_string.split(",") if t.strip()][:n]
    parts = []
    for i, t in enumerate(tags):
        fg, bg = TAG_STYLES[i % len(TAG_STYLES)]
        parts.append(f'<span class="tag" style="color:{fg};background:{bg}">{esc(t)}</span>')
    return "".join(parts)


PREVIEW_CHARS = 260
BOILERPLATE = re.compile(r"There is an? (?:alternate|other) cover edition[^.]*\.\s*", re.I)


def clean_desc(text):
    return BOILERPLATE.sub("", str(text)).strip()


def preview(text, n=PREVIEW_CHARS):
    text = clean_desc(text).rstrip("…").strip()
    if len(text) <= n:
        return text
    return text[:n].rsplit(" ", 1)[0].rstrip(".,;:!- ") + "…"


def desc_html(text):
    full = clean_desc(text)
    short = preview(full)
    if len(short) >= len(full):
        return f'<div class="b-desc">{esc(full)}</div>'
    return (
        '<details class="desc b-desc"><summary>'
        f'<span class="d-short">{esc(short)} <span class="see">See more</span></span>'
        f'<span class="d-full">{esc(full)} <span class="see">See less</span></span>'
        "</summary></details>"
    )


def cover_html(row):
    url = row["cover_url"]
    if isinstance(url, str) and url.strip():
        return f'<img class="cover" src="{esc(url)}" referrerpolicy="no-referrer" alt="Book cover">'
    return f'<div class="cover cover-ph">{esc(row["title"][:45])}</div>'


from similar_rules import bundle_mask, pick_similar, series_keys

# Precomputed once: who wrote each book, which series it is in, which are boxed sets, which have reviews
_author_ids = (books_df["author_id"].astype(str).tolist() if "author_id" in books_df.columns
               else [""] * len(books_df))
if "series_keys" in books_df.columns:
    _series_sets = [frozenset(k) for k in books_df["series_keys"]]
else:  # older data file: use only the series name written in the title
    _series_sets = [series_keys([], t) for t in books_df["title"]]
_is_bundle = bundle_mask(books_df["title"])
_has_reviews = books_df["book_id"].isin(sent_lookup.index).to_numpy()


def similar_indices(index, n=5):
    scores = cosine_similarity(tfidf_matrix[index], tfidf_matrix).ravel()
    # The rules for choosing books are in similar_rules.py. 
    return pick_similar(scores, index, _author_ids, _series_sets, _is_bundle, _has_reviews, n=n)


def tier_info(book_id):
    if book_id in sent_lookup.index:
        r = sent_lookup.loc[book_id]
        return r["tier"], int(r["n_reviews"]), float(r["pos_pct"])
    return "No reviews", 0, 0.0


# ---------------------------------------------------------------
# Session memory
# ---------------------------------------------------------------
for key, default in [("staff_picks", []), ("featured", []), ("show_recs", False), ("rec_idx", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------
# Top bar
# ---------------------------------------------------------------
picked = st.session_state.staff_picks
window = st.session_state.featured

st.markdown(
    f'<div class="topbar">'
    f'<div class="brand"><div class="brand-logo">📚</div>'
    f'<div><div class="brand-name">ReadWell Staff Console</div>'
    f'<div class="tagline">Find similar books and see what readers think. A simple tool for bookshop and library staff.</div></div></div>'
    f"</div>",
    unsafe_allow_html=True,
)


def show_list(kind, items, empty_text):
    """One list (staff picks or window), with a Remove button for each book."""
    if not items:
        st.caption(empty_text)
        return
    for n, title in enumerate(items):
        col_name, col_btn = st.columns([5, 1])
        col_name.text(f"{n + 1}. {title}")
        if col_btn.button("Remove", key=f"remove_{kind}_{n}"):
            items.pop(n)
            st.rerun()

# ---------------------------------------------------------------
# Choose a book
# ---------------------------------------------------------------
st.markdown(
    '<div class="hero"><h2>Discover similar books</h2>'
    "<p>Choose a genre and a book to get 5 similar books, with how readers felt about them.</p></div>",
    unsafe_allow_html=True,
)

# two separate lists, one tab each (they only live for this visit; nothing is saved)
with st.expander(f"Your lists:  Staff picks ({len(picked)})  |  In the window ({len(window)})"):
    tab_picks, tab_window = st.tabs([f"Staff picks ({len(picked)})", f"In the window ({len(window)})"])
    with tab_picks:
        show_list("pick", picked, "No staff picks yet. Staff picks are books that staff recommend to customers. Use “Add to staff picks” on a book below.")
    with tab_window:
        show_list("window", window, "No books in the window yet. The window display is where books are shown to attract attention. Use “Feature in window” on a book below.")

GENRES = ["All genres", "Romance", "Fantasy", "Mystery"]
col_genre, col_book = st.columns([1, 2])
genre = col_genre.selectbox("Select a genre", GENRES)

if genre == "All genres":
    pool = books_df
else:
    pool = books_df[books_df["genres"].map(lambda g: genre.lower() in g)]

if len(pool) == 0:
    st.warning("No books found for this genre.")
    st.stop()

pool = pool.sort_values("ratings_count", ascending=False)  # most popular books first


def book_label(i):
    row = books_df.loc[i]
    author = f" — {row['author']}" if row["author"] else ""
    return f"{row['title']}{author}"


selected_idx = col_book.selectbox("Select a book (type to search)", pool.index.tolist(), format_func=book_label)

chosen = books_df.loc[selected_idx]
c1, c2 = st.columns([1, 6])
c1.markdown(cover_html(chosen), unsafe_allow_html=True)
c2.markdown(
    f'<div class="b-title">{esc(chosen["title"])}</div>'
    f'<div class="b-author">{esc(chosen["author"])}</div>'
    f'{tags_html(chosen["clean_tags_str"], 5)}',
    unsafe_allow_html=True,
)

st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)  # space between the cover and the button
if st.button("Get recommendations", type="primary"):
    st.session_state.show_recs = True
    st.session_state.rec_idx = selected_idx

# ---------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------
if st.session_state.show_recs and st.session_state.rec_idx is not None:
    base = books_df.loc[st.session_state.rec_idx]
    st.markdown('<div class="section-title">Similar titles for staff to consider</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-sub">These books are similar to <b>{esc(base["title"])}</b> based on their content, '
        f"and show how readers felt about them.</div>",
        unsafe_allow_html=True,
    )

    for rank, i in enumerate(similar_indices(st.session_state.rec_idx), start=1):
        row = books_df.iloc[i]
        bid = row["book_id"]
        tier, n_reviews, pos_pct = tier_info(bid)
        label, fg, bg, mark = TIERS.get(tier, TIERS["No reviews"])
        review_text = f"Based on {n_reviews:,} reviews" if n_reviews else "No reviews in the data"
        year = f"{int(row['year'])}" if pd.notna(row["year"]) and row["year"] else ""

        with st.container(border=True):
            st.markdown('<span class="card-marker"></span>', unsafe_allow_html=True)
            col_img, col_main, col_side = st.columns([1.1, 4.2, 2.3])

            col_img.markdown(cover_html(row), unsafe_allow_html=True)

            col_main.markdown(
                f'<div class="rank">{rank}</div>'
                f'<div class="b-title">{esc(row["title"])}</div>'
                f'<div class="b-author">{esc(row["author"])}</div>'
                f'{tags_html(row["clean_tags_str"], 3)}'
                f'{desc_html(row["short_desc"])}',
                unsafe_allow_html=True,
            )

            col_side.markdown(
                f'<div class="sent" style="background:{bg};color:{fg}">'
                f'<div class="sent-title">{mark} {label}</div><div class="sent-sub">{review_text}</div></div>',
                unsafe_allow_html=True,
            )
            if col_side.button("Add to staff picks", key=f"pick_{bid}",
                               help="Staff picks: a short list of books that staff choose to recommend to customers, based on trends, popularity and customer requests."):
                if row["title"] not in st.session_state.staff_picks:
                    st.session_state.staff_picks.append(row["title"])
                    st.toast(f"Added '{row['title']}' to staff picks")
                    st.rerun()
            if col_side.button("Feature in window", key=f"feature_{bid}",
                               help="Window display: books picked separately for the window or a display table to attract attention, based on popularity or recommendations."):
                if row["title"] not in st.session_state.featured:
                    st.session_state.featured.append(row["title"])
                    st.toast(f"Added '{row['title']}' to the window display")
                    st.rerun()

            with st.expander("View details"):
                if n_reviews:
                    st.write(
                        f"**{label}.** The model reads about **{round(pos_pct)}%** of this book's "
                        f"{n_reviews:,} reviews as positive. This is an estimate, not an exact count, "
                        "and it tends to run a little low."
                    )
                else:
                    st.write("There are no reviews for this book in the data.")
                st.write(
                    f"Rated by {int(row['ratings_count']):,} Goodreads readers"
                    + (f" · this edition published {year}" if year else "")
                )
                st.markdown(tags_html(row["clean_tags_str"], 8), unsafe_allow_html=True)

# ---------------------------------------------------------------
# Footer note
# ---------------------------------------------------------------
st.markdown(
    '<div class="note"><b>About the recommendations</b><br>'
    "Books are matched by their content (description and reader tags). The label shows how readers felt, "
    "based on Goodreads reviews, compared with the other books in the dataset. It is a guide, not an exact score.<br><br>"
    "<b>Prototype note:</b> this demo only contains popular books from three genres (romance, fantasy and mystery), "
    "taken from a Goodreads dataset collected several years ago. Very recent books are not included.</div>",
    unsafe_allow_html=True,
)

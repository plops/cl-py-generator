//! mapserver: Standalone-Kartenvalidierung für das 174_autocluster-Clustering.
//! Daten-Layer: CSV/JSON laden, validieren, read-only DB-Lookup. Kein Volltext in
//! Bulk-Strukturen — Anti-Scraping per Design (s. plan.md D4/D5, Abschnitt 4).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::Serialize;

// Provenienz (Quellen: doe/PHASEB_de.md, TITLING-meta, Walkthrough Abschnitt 8).
pub const METHOD: &str = "HDBSCAN (mcs=11, ms=8) auf UMAP (d=11, nn=45, md=0,09, k=3072)";
pub const CLUSTERING_DATE: &str = "2026-09-27";
pub const METHODIK_REF: &str = "plan/20260927_01_review_doe/walkthrough.md (Abschnitt 8)";
pub const NOISE_TITLE: &str = "Noise";
pub const DEFAULT_DETAIL_PER_MIN: u32 = 60;

/// Laufzeit-Konfiguration aus ENV mit sinnvollen Defaults (CWD = Repo-Ordner
/// `174_autocluster`, d. h. `cargo run` aus `mapserver/` nutzt `../`-Pfade).
#[derive(Debug, Clone)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub db_path: PathBuf,
    pub coords_path: PathBuf,
    pub labels_path: PathBuf,
    pub titles_path: PathBuf,
    pub detail_per_min: u32,
}

fn env(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        let port: u16 = env("PORT", "8080")
            .parse()
            .map_err(|e| anyhow::anyhow!("PORT ungültig: {e}"))?;
        let detail_per_min: u32 = env("MAP_DETAIL_PER_MIN", "60")
            .parse()
            .map_err(|e| anyhow::anyhow!("MAP_DETAIL_PER_MIN ungültig: {e}"))?;
        Ok(Self {
            host: env("HOST", "127.0.0.1"),
            port,
            db_path: env("MAP_DB", "../summaries_compact_20260924.db").into(),
            coords_path: env("MAP_COORDS", "../plots/coords_phaseb_2d.csv").into(),
            labels_path: env("MAP_LABELS", "../plots/labels_phaseb.csv").into(),
            titles_path: env("MAP_TITLES", "../cluster_titles_phaseb.json").into(),
            detail_per_min: if detail_per_min == 0 {
                DEFAULT_DETAIL_PER_MIN
            } else {
                detail_per_min
            },
        })
    }
}

/// Ein Kartenpunkt — bewusst ohne Textfeld (Anti-Scraping).
#[derive(Debug, Clone, Serialize)]
pub struct Point {
    pub identifier: i64,
    pub x: f32,
    pub y: f32,
    pub cluster: i32,
}

/// Cluster für Legende/API — nur Titel + Größe, nie Text.
#[derive(Debug, Clone, Serialize)]
pub struct ClusterInfo {
    pub id: i32,
    pub title: String,
    pub n: usize,
}

#[derive(Debug, Clone)]
pub struct AppData {
    pub points: Vec<Point>,
    pub clusters: Vec<ClusterInfo>,
    pub noise_n: usize,
}

impl AppData {
    pub fn title_of(&self, cluster: i32) -> &str {
        if cluster == -1 {
            return NOISE_TITLE;
        }
        self.clusters
            .iter()
            .find(|c| c.id == cluster)
            .map(|c| c.title.as_str())
            .unwrap_or("?")
    }
}

/// Detailantwort: genau EIN Summary-Text (einzige Stelle im System).
#[derive(Debug, Clone, Serialize)]
pub struct PointDetail {
    pub identifier: i64,
    pub cluster: i32,
    pub cluster_title: String,
    pub summary: String,
    pub original_source_link: String,
}

/// Zeile tolerant splitten: Header sind kommagetrennt, Datenzeilen
/// leerzeichengetrennt (Quirk der Eingabedaten, per `od` verifiziert).
fn split_row(line: &str) -> Vec<&str> {
    line.split([',', ' ', '\t'])
        .filter(|s| !s.is_empty())
        .collect()
}

fn parse_coords(text: &str) -> anyhow::Result<Vec<(i64, f32, f32)>> {
    let mut lines = text.lines();
    let header = split_row(lines.next().unwrap_or(""));
    anyhow::ensure!(header == ["identifier", "x", "y"], "coords-Header unerwartet");
    lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let p = split_row(l);
            anyhow::ensure!(p.len() == 3, "coords-Zeile ungültig: {l}");
            Ok((p[0].parse()?, p[1].parse()?, p[2].parse()?))
        })
        .collect()
}

fn parse_labels(text: &str) -> anyhow::Result<HashMap<i64, i32>> {
    let mut lines = text.lines();
    let header = split_row(lines.next().unwrap_or(""));
    anyhow::ensure!(header == ["identifier", "cluster"], "labels-Header unerwartet");
    let mut map = HashMap::new();
    for l in lines.filter(|l| !l.trim().is_empty()) {
        let p = split_row(l);
        anyhow::ensure!(p.len() == 2, "labels-Zeile ungültig: {l}");
        map.insert(p[0].parse()?, p[1].parse()?);
    }
    Ok(map)
}

#[derive(serde::Deserialize)]
struct TitlesFile {
    titles: HashMap<String, TitleEntry>,
}

#[derive(serde::Deserialize)]
struct TitleEntry {
    title: String,
    n: usize, // members/exemplars/etc. werden ignoriert (nie laden!)
}

fn parse_titles(text: &str) -> anyhow::Result<HashMap<i32, (String, usize)>> {
    let file: TitlesFile = serde_json::from_str(text)?;
    file.titles
        .into_iter()
        .map(|(k, v)| Ok((k.parse()?, (v.title, v.n))))
        .collect()
}

/// Join + Validierung: ID-Mengen identisch, Label-IDs == Titel-IDs (+ Noise),
/// Titel-Counts == Label-Counts. Jeder Verstoß ist ein Startfehler (laut statt
/// falsch — eine Karte mit stillen Inkonsistenzen wäre irreführend).
pub fn load_app_data(
    coords_path: &Path,
    labels_path: &Path,
    titles_path: &Path,
) -> anyhow::Result<AppData> {
    let coords = parse_coords(&std::fs::read_to_string(coords_path)?)?;
    let labels = parse_labels(&std::fs::read_to_string(labels_path)?)?;
    let titles = parse_titles(&std::fs::read_to_string(titles_path)?)?;
    join(coords, &labels, &titles)
}

fn join(
    coords: Vec<(i64, f32, f32)>,
    labels: &HashMap<i64, i32>,
    titles: &HashMap<i32, (String, usize)>,
) -> anyhow::Result<AppData> {
    anyhow::ensure!(!coords.is_empty(), "keine Koordinaten");
    let coord_ids: HashSet<i64> = coords.iter().map(|(id, _, _)| *id).collect();
    let label_ids: HashSet<i64> = labels.keys().copied().collect();
    anyhow::ensure!(coord_ids == label_ids, "ID-Mengen coords/labels weichen ab");

    let mut points = Vec::with_capacity(coords.len());
    let mut counts: HashMap<i32, usize> = HashMap::new();
    for (id, x, y) in coords {
        let cluster = labels[&id];
        *counts.entry(cluster).or_default() += 1;
        points.push(Point {
            identifier: id,
            x,
            y,
            cluster,
        });
    }
    points.sort_by_key(|p| p.identifier);

    let mut label_clusters: HashSet<i32> = counts.keys().copied().collect();
    label_clusters.remove(&-1);
    let title_ids: HashSet<i32> = titles.keys().copied().collect();
    anyhow::ensure!(label_clusters == title_ids, "Label-IDs ungleich Titel-IDs");

    let mut clusters: Vec<ClusterInfo> = titles
        .iter()
        .map(|(id, (title, n))| {
            anyhow::ensure!(
                counts.get(id).copied().unwrap_or(0) == *n,
                "Count-Mismatch bei Cluster {id}"
            );
            Ok(ClusterInfo {
                id: *id,
                title: title.clone(),
                n: *n,
            })
        })
        .collect::<anyhow::Result<_>>()?;
    clusters.sort_by_key(|c| c.id);

    Ok(AppData {
        points,
        clusters,
        noise_n: counts.get(&-1).copied().unwrap_or(0),
    })
}

/// (Requests/Sekunde, Burst) aus „pro Minute": Burst erlaubt kurze Serien
/// (z. B. neugieriges Durchklicken), die Rate begrenzt den Durchschnitt.
pub fn quota_for_detail(per_min: u32) -> (u64, u32) {
    let per_min = if per_min == 0 {
        DEFAULT_DETAIL_PER_MIN
    } else {
        per_min
    };
    (std::cmp::max(1, (per_min / 60) as u64), per_min)
}

/// Genau eine DB-Zeile lesen — synchron, daher nur aus `spawn_blocking`.
/// Read-only-Öffnung ist Pflicht (ENV.md); `None` = unbekannte ID → 404.
pub fn fetch_detail_row(db_path: &Path, identifier: i64) -> anyhow::Result<Option<(String, String)>> {
    let conn = rusqlite::Connection::open_with_flags(
        db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )?;
    let mut stmt = conn.prepare(
        "SELECT summary, original_source_link FROM summaries WHERE identifier = ?",
    )?;
    let mut rows = stmt.query([identifier])?;
    match rows.next()? {
        None => Ok(None),
        Some(row) => {
            let summary: Option<String> = row.get(0)?;
            let link: Option<String> = row.get(1)?;
            Ok(Some((summary.unwrap_or_default(), link.unwrap_or_default())))
        }
    }
}

// ---- Web-Layer (Axum; Muster aus rs-summarizer: build_router + State) ----

use axum::{
    Router,
    extract::{Path as UrlPath, State},
    http::{StatusCode, header},
    response::{IntoResponse, Json},
    routing::get,
};
use std::sync::Arc;
use tower_governor::{GovernorLayer, governor::GovernorConfigBuilder};

#[derive(Debug, Clone)]
pub struct AppState {
    pub data: Arc<AppData>,
    pub db_path: PathBuf,
}

/// Router bauen. Das Rate-Limit hängt NUR an der Detail-Route (dort liegt der
/// einzige Volltext); Bulk-Routen brauchen keines, weil sie nie Text liefern.
pub fn build_router(state: AppState, detail_per_min: u32) -> Router {
    let (per_sec, burst) = quota_for_detail(detail_per_min);
    let detail_conf = GovernorConfigBuilder::default()
        .per_second(per_sec)
        .burst_size(burst)
        .finish()
        .expect("Governor-Quota ungültig");
    let limited = Router::new()
        .route("/api/map/point/{id}", get(point_detail))
        .layer(GovernorLayer::new(detail_conf));
    Router::new()
        .route("/healthz", get(health))
        .route("/api/map/points", get(points))
        .route("/api/map/clusters", get(clusters))
        .merge(limited)
        .with_state(state)
}

async fn health(State(st): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "points": st.data.points.len(),
        "clusters": st.data.clusters.len(),
    }))
}

/// Punkte sind pro Deployment statisch → lang cachen.
const CACHE_STATIC: &str = "public, max-age=86400";

async fn points(State(st): State<AppState>) -> impl IntoResponse {
    ([(header::CACHE_CONTROL, CACHE_STATIC)], Json(st.data.points.clone()))
}

async fn clusters(State(st): State<AppState>) -> impl IntoResponse {
    let mut list = st.data.clusters.clone();
    list.push(ClusterInfo {
        id: -1,
        title: NOISE_TITLE.into(),
        n: st.data.noise_n,
    });
    ([(header::CACHE_CONTROL, CACHE_STATIC)], Json(list))
}

async fn point_detail(State(st): State<AppState>, UrlPath(id): UrlPath<i64>) -> impl IntoResponse {
    let idx = match st.data.points.binary_search_by_key(&id, |p| p.identifier) {
        Ok(i) => i,
        Err(_) => return (StatusCode::NOT_FOUND, "unbekannter Punkt").into_response(),
    };
    let cluster = st.data.points[idx].cluster;
    let db_path = st.db_path.clone();
    let row = tokio::task::spawn_blocking(move || fetch_detail_row(&db_path, id)).await;
    let (summary, link) = match row {
        Ok(Ok(Some((s, l)))) => (s, l),
        Ok(Ok(None)) => {
            return (StatusCode::NOT_FOUND, "kein Detail vorhanden").into_response();
        }
        _ => return (StatusCode::INTERNAL_SERVER_ERROR, "DB-Fehler").into_response(),
    };
    Json(PointDetail {
        identifier: id,
        cluster,
        cluster_title: st.data.title_of(cluster).to_string(),
        summary,
        original_source_link: link,
    })
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    const COORDS: &str = "identifier,x,y\n1 5.1 6.0\n2 4.8 6.4\n3 4.9 6.5\n";
    const LABELS: &str = "identifier,cluster\n1 -1\n2 0\n3 0\n";
    const TITLES: &str = r#"{"version":1,"meta":{},"titles":{"0":{"title":"Titel Null","n":2,"member_sig":"x","job_sig":"y","members":[2,3],"exemplars":[2],"neighbors":[]}}}"#;

    fn fixture() -> AppData {
        join(
            parse_coords(COORDS).unwrap(),
            &parse_labels(LABELS).unwrap(),
            &parse_titles(TITLES).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn quirk_csv_wird_tolerant_geparst() {
        let coords = parse_coords(COORDS).unwrap();
        assert_eq!(coords.len(), 3);
        assert_eq!(coords[0], (1, 5.1, 6.0));
        let labels = parse_labels(LABELS).unwrap();
        assert_eq!(labels[&1], -1);
        assert_eq!(labels[&3], 0);
    }

    #[test]
    fn join_validiert_counts_titel_noise() {
        let data = fixture();
        assert_eq!(data.points.len(), 3);
        assert_eq!(data.points[0].identifier, 1); // sortiert
        assert_eq!(data.clusters.len(), 1);
        assert_eq!(data.clusters[0].title, "Titel Null"); // members ignoriert
        assert_eq!(data.noise_n, 1);
        assert_eq!(data.title_of(-1), "Noise");
        assert_eq!(data.title_of(0), "Titel Null");
    }

    #[test]
    fn id_mismatch_ist_startfehler() {
        let bad_labels = parse_labels("identifier,cluster\n1 -1\n2 0\n4 0\n").unwrap();
        let err = join(
            parse_coords(COORDS).unwrap(),
            &bad_labels,
            &parse_titles(TITLES).unwrap(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("ID-Mengen"));
    }

    #[test]
    fn quota_mapping() {
        assert_eq!(quota_for_detail(60), (1, 60));
        assert_eq!(quota_for_detail(2), (1, 2));
        assert_eq!(quota_for_detail(0), (1, 60)); // 0 → Default
        assert_eq!(quota_for_detail(120), (2, 120));
    }
}

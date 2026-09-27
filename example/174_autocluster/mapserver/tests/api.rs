//! Integration-Tests gegen einen echten Fixture-Server (127.0.0.1:0, rohes HTTP).
//! Beweist u. a. Anti-Scraping: Bulk-APIs liefern nie Volltext.

use std::net::SocketAddr;
use std::sync::Arc;

use mapserver::{AppState, build_router, load_app_data};

const COORDS: &str = "identifier,x,y\n1 1.0 2.0\n2 3.0 4.0\n3 5.0 6.0\n4 7.0 8.0\n";
const LABELS: &str = "identifier,cluster\n1 -1\n2 0\n3 0\n4 1\n";
const TITLES: &str = r#"{"version":1,"meta":{},"titles":{"0":{"title":"Cluster Null","n":2},"1":{"title":"Cluster Eins","n":1}}}"#;
/// Marker im Fixture-Volltext: darf in KEINER Bulk-Antwort auftauchen.
const SECRET: &str = "SUMMARY-GEHEIM";

fn write_fixtures(dir: &std::path::Path) {
    std::fs::write(dir.join("coords.csv"), COORDS).unwrap();
    std::fs::write(dir.join("labels.csv"), LABELS).unwrap();
    std::fs::write(dir.join("titles.json"), TITLES).unwrap();
    let conn = rusqlite::Connection::open(dir.join("fix.db")).unwrap();
    conn.execute(
        "CREATE TABLE summaries(identifier INTEGER PRIMARY KEY, summary TEXT, original_source_link TEXT)",
        [],
    )
    .unwrap();
    for id in 1..=4 {
        conn.execute(
            "INSERT INTO summaries VALUES (?, ?, ?)",
            rusqlite::params![
                id,
                format!("{SECRET}-{id} lies hier"),
                format!("https://example.com/video-{id}")
            ],
        )
        .unwrap();
    }
}

/// Fixture-Server starten; TempDir-Guard am Leben halten (DB-Datei).
async fn spawn_server(per_min: u32) -> (u16, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    write_fixtures(tmp.path());
    let data = load_app_data(
        &tmp.path().join("coords.csv"),
        &tmp.path().join("labels.csv"),
        &tmp.path().join("titles.json"),
    )
    .unwrap();
    let state = AppState {
        data: Arc::new(data),
        db_path: tmp.path().join("fix.db"),
    };
    let app = build_router(state, per_min);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        .unwrap();
    });
    (port, tmp)
}

/// Rohes HTTP-GET ohne extra Client-Dep; liefert (Status, Body).
async fn get(port: u16, path: &str) -> (u16, String, String) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut stream = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    stream
        .write_all(
            format!("GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n").as_bytes(),
        )
        .await
        .unwrap();
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).await.unwrap();
    let text = String::from_utf8_lossy(&buf).into_owned();
    let mut parts = text.splitn(2, "\r\n\r\n");
    let head = parts.next().unwrap_or("").to_string();
    let body = parts.next().unwrap_or("").to_string();
    let status: u16 = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    (status, head, body)
}

fn keys_of(obj: &serde_json::Value) -> Vec<String> {
    let mut keys: Vec<String> = obj.as_object().unwrap().keys().cloned().collect();
    keys.sort();
    keys
}

#[tokio::test]
async fn health_meldet_counts() {
    let (port, _tmp) = spawn_server(60).await;
    let (status, _, body) = get(port, "/healthz").await;
    assert_eq!(status, 200);
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["status"], "ok");
    assert_eq!(v["points"], 4);
    assert_eq!(v["clusters"], 2);
}

#[tokio::test]
async fn punkte_api_ohne_textfeld() {
    let (port, _tmp) = spawn_server(60).await;
    let (status, head, body) = get(port, "/api/map/points").await;
    assert_eq!(status, 200);
    assert!(head.to_lowercase().contains("cache-control"));
    assert!(!body.contains(SECRET), "Volltext-Leak in Bulk-API!");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let arr = v.as_array().unwrap();
    assert_eq!(arr.len(), 4);
    for p in arr {
        assert_eq!(keys_of(p), ["cluster", "identifier", "x", "y"]);
    }
    assert_eq!(arr[0]["identifier"], 1);
    assert_eq!(arr[0]["cluster"], -1);
}

#[tokio::test]
async fn cluster_api_mit_noise_zeile() {
    let (port, _tmp) = spawn_server(60).await;
    let (status, _, body) = get(port, "/api/map/clusters").await;
    assert_eq!(status, 200);
    assert!(!body.contains(SECRET));
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let arr = v.as_array().unwrap();
    assert_eq!(arr.len(), 3); // 2 Cluster + Noise
    for c in arr {
        assert_eq!(keys_of(c), ["id", "n", "title"]);
    }
    let noise = arr.iter().find(|c| c["id"] == -1).unwrap();
    assert_eq!(noise["title"], "Noise");
    assert_eq!(noise["n"], 1);
}

#[tokio::test]
async fn detail_api_genau_ein_text() {
    let (port, _tmp) = spawn_server(60).await;
    let (status, _, body) = get(port, "/api/map/point/2").await;
    assert_eq!(status, 200);
    assert_eq!(body.matches(SECRET).count(), 1, "genau ein Summary-Text");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        keys_of(&v),
        ["cluster", "cluster_title", "identifier", "original_source_link", "summary"]
    );
    assert_eq!(v["identifier"], 2);
    assert_eq!(v["cluster_title"], "Cluster Null");
    assert!(v["original_source_link"].as_str().unwrap().contains("example.com"));
}

#[tokio::test]
async fn detail_api_404_bei_unbekannt() {
    let (port, _tmp) = spawn_server(60).await;
    let (status, _, _) = get(port, "/api/map/point/999").await;
    assert_eq!(status, 404);
}

#[tokio::test]
async fn karten_shell_ohne_volltext() {
    let (port, _tmp) = spawn_server(60).await;
    let (status, _, body) = get(port, "/map").await;
    assert_eq!(status, 200);
    assert!(body.contains("id=\"karte\""));
    assert!(body.contains("Cluster-Karte"));
    assert!(body.contains("Methodik"));
    assert!(body.contains("zeigeDetail"), "Detail-Logik fehlt in Shell!");
    assert!(body.contains("URLSearchParams"), "Deep-Link (?point=) fehlt!");
    assert!(!body.contains(SECRET), "Volltext-Leak in Shell!");
}

#[tokio::test]
async fn root_leitet_auf_karte() {
    let (port, _tmp) = spawn_server(60).await;
    let (status, head, _) = get(port, "/").await;
    assert_eq!(status, 308);
    assert!(head.contains("/map"));
}

#[tokio::test]
async fn rate_limit_greift() {
    let (port, _tmp) = spawn_server(1).await; // Burst 1
    let (s1, _, _) = get(port, "/api/map/point/2").await;
    let (s2, _, _) = get(port, "/api/map/point/2").await;
    assert_eq!((s1, s2), (200, 429));
}

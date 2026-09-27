use std::net::SocketAddr;
use std::sync::Arc;

use mapserver::{AppState, Config, build_router, check_db, load_app_data};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cfg = Config::from_env()?;
    let data = load_app_data(&cfg.coords_path, &cfg.labels_path, &cfg.titles_path)
        .map_err(|e| anyhow::anyhow!("{e} (ENV MAP_COORDS/MAP_LABELS/MAP_TITLES prüfen)"))?;
    // DB nur warnen, nicht abbrechen: Karte bleibt ansehbar, Klicks melden
    // den Fehler im Panel (db_ok=false steht auch in /healthz).
    let (db_ok, db_rows) = match check_db(&cfg.db_path) {
        Ok(n) => (true, n),
        Err(e) => {
            tracing::warn!(
                "DB-Prüfung fehlgeschlagen: {e} — ENV MAP_DB (aktuell: {}) prüfen! \
                 Karte startet trotzdem, aber Klicks liefern keine Details.",
                cfg.db_path.display()
            );
            (false, 0)
        }
    };
    let state = AppState {
        data: Arc::new(data),
        db_path: cfg.db_path.clone(),
        db_ok,
        db_rows,
    };
    tracing::info!(
        points = state.data.points.len(),
        clusters = state.data.clusters.len(),
        noise = state.data.noise_n,
        db_rows,
        "Kartendaten geladen"
    );
    let app = build_router(state, cfg.detail_per_min);
    let addr: SocketAddr = format!("{}:{}", cfg.host, cfg.port)
        .parse()
        .map_err(|e| anyhow::anyhow!("Adresse ungültig: {e}"))?;
    tracing::info!("mapserver hört auf {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await?;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

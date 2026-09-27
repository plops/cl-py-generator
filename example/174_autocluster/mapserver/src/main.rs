use std::net::SocketAddr;
use std::sync::Arc;

use mapserver::{AppState, Config, build_router, load_app_data};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cfg = Config::from_env()?;
    let data = load_app_data(&cfg.coords_path, &cfg.labels_path, &cfg.titles_path)?;
    let state = AppState {
        data: Arc::new(data),
        db_path: cfg.db_path.clone(),
    };
    tracing::info!(
        points = state.data.points.len(),
        clusters = state.data.clusters.len(),
        noise = state.data.noise_n,
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

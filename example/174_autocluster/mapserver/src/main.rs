use mapserver::{Config, load_app_data};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cfg = Config::from_env()?;
    let data = load_app_data(&cfg.coords_path, &cfg.labels_path, &cfg.titles_path)?;
    tracing::info!(
        points = data.points.len(),
        clusters = data.clusters.len(),
        noise = data.noise_n,
        port = cfg.port,
        "Kartendaten geladen (Server folgt in S2)"
    );
    Ok(())
}

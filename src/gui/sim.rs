use crate::backend::p2p::{Contract, P2PNetwork, P2PNode};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::{Backend, CrosstermBackend},
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, List, ListItem, Paragraph, Row, Table, Tabs},
    Frame, Terminal,
};
use std::{
    io,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

pub enum SimulationSpeed {
    Paused,
    Slow,
    Normal,
    Fast,
}

impl SimulationSpeed {
    fn as_millis(&self) -> Option<u64> {
        match self {
            SimulationSpeed::Paused => None,
            SimulationSpeed::Slow => Some(500),
            SimulationSpeed::Normal => Some(100),
            SimulationSpeed::Fast => Some(10),
        }
    }

    fn to_string(&self) -> &str {
        match self {
            SimulationSpeed::Paused => "Paused",
            SimulationSpeed::Slow => "Slow",
            SimulationSpeed::Normal => "Normal",
            SimulationSpeed::Fast => "Fast",
        }
    }
}

pub struct App {
    tab_index: usize,
    network: Arc<Mutex<P2PNetwork>>,
    simulation_speed: SimulationSpeed,
    running: Arc<AtomicBool>,
    selected_node: Option<usize>,
}

impl App {
    pub fn new(network: P2PNetwork) -> Self {
        App {
            tab_index: 0,
            network: Arc::new(Mutex::new(network)),
            simulation_speed: SimulationSpeed::Normal,
            running: Arc::new(AtomicBool::new(true)),
            selected_node: None,
        }
    }

    pub fn run_simulation(&mut self) -> io::Result<()> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        // Simulation thread
        let network_clone = Arc::clone(&self.network);
        let running_clone = Arc::clone(&self.running);
        let simulation_handle = thread::spawn(move || {
            let mut iter = 0;
            while running_clone.load(Ordering::SeqCst) {
                // Get current speed
                {
                    let mut network = network_clone.lock().unwrap();
                    let graph = P2PNetwork::make_graph(&network);
                    let order = graph.topological_sort().unwrap_or_default();

                    // Process nodes without contracts
                    let nodes_to_process = P2PNetwork::nodes_without_contracts(&network);

                    // Create new contracts
                    let mut new_contracts = Vec::new();
                    for node_info in nodes_to_process {
                        if let Ok(path) =
                            P2PNetwork::fastest_path_for_node(&network, node_info, &graph, &order)
                        {
                            if let Ok(contract) =
                                P2PNetwork::create_contract(&network, node_info, path)
                            {
                                new_contracts.push(contract);
                            }
                        }
                    }

                    // Update network state
                    let node_to_update = iter as usize % network.nodes.len();
                    P2PNetwork::update_network(&mut network, 1, iter, 1000, node_to_update);

                    // Add all new contracts at once
                    network.contracts.extend(new_contracts);
                }

                iter += 1;
                thread::sleep(Duration::from_millis(100));
            }
        });

        // Main rendering loop
        let mut last_tick = Instant::now();
        let tick_rate = Duration::from_millis(100);

        loop {
            terminal.draw(|f| self.ui(f))?;

            let timeout = tick_rate
                .checked_sub(last_tick.elapsed())
                .unwrap_or_else(|| Duration::from_secs(0));

            if crossterm::event::poll(timeout)? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Char('q') => break,
                        KeyCode::Tab => {
                            self.tab_index = (self.tab_index + 1) % 3;
                        }
                        KeyCode::Char(' ') => {
                            self.toggle_pause();
                        }
                        KeyCode::Right => {
                            self.increase_speed();
                        }
                        KeyCode::Left => {
                            self.decrease_speed();
                        }
                        KeyCode::Up => {
                            if let Some(idx) = self.selected_node {
                                if idx > 0 {
                                    self.selected_node = Some(idx - 1);
                                }
                            } else {
                                self.selected_node = Some(0);
                            }
                        }
                        KeyCode::Down => {
                            let network = self.network.lock().unwrap();
                            let max_idx = network.nodes.len().saturating_sub(1);
                            if let Some(idx) = self.selected_node {
                                if idx < max_idx {
                                    self.selected_node = Some(idx + 1);
                                }
                            } else {
                                self.selected_node = Some(0);
                            }
                        }
                        _ => {}
                    }
                }
            }

            if last_tick.elapsed() >= tick_rate {
                last_tick = Instant::now();
            }
        }

        // Cleanup
        self.running.store(false, Ordering::SeqCst);
        simulation_handle.join().unwrap();

        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;

        Ok(())
    }

    fn toggle_pause(&mut self) {
        self.simulation_speed = match self.simulation_speed {
            SimulationSpeed::Paused => SimulationSpeed::Normal,
            _ => SimulationSpeed::Paused,
        };
    }

    fn increase_speed(&mut self) {
        self.simulation_speed = match self.simulation_speed {
            SimulationSpeed::Paused => SimulationSpeed::Slow,
            SimulationSpeed::Slow => SimulationSpeed::Normal,
            SimulationSpeed::Normal => SimulationSpeed::Fast,
            SimulationSpeed::Fast => SimulationSpeed::Fast,
        };
    }

    fn decrease_speed(&mut self) {
        self.simulation_speed = match self.simulation_speed {
            SimulationSpeed::Paused => SimulationSpeed::Paused,
            SimulationSpeed::Slow => SimulationSpeed::Paused,
            SimulationSpeed::Normal => SimulationSpeed::Slow,
            SimulationSpeed::Fast => SimulationSpeed::Normal,
        };
    }

    fn ui<B: Backend>(&self, f: &mut Frame<B>) {
        let size = f.size();

        // Create layout sections
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints(
                [
                    Constraint::Length(3),
                    Constraint::Min(5),
                    Constraint::Length(3),
                ]
                .as_ref(),
            )
            .split(size);

        // Create tabs
        let titles = ["Network Nodes", "Active Contracts", "Statistics"]
            .iter()
            .map(|t| {
                let (first, rest) = t.split_at(1);
                Line::from(vec![
                    Span::styled(
                        first,
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::UNDERLINED),
                    ),
                    Span::styled(rest, Style::default().fg(Color::White)),
                ])
            })
            .collect();

        let tabs = Tabs::new(titles)
            .block(
                Block::default()
                    .title("P2P Network Simulator")
                    .borders(Borders::ALL),
            )
            .select(self.tab_index)
            .style(Style::default().fg(Color::White))
            .highlight_style(
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            );

        f.render_widget(tabs, chunks[0]);

        // Status bar at the bottom
        let status_text = format!(
            "Speed: {} | [←/→]: Adjust Speed | [Space]: Pause/Resume | [Tab]: Change View | [q]: Quit",
            self.simulation_speed.to_string()
        );
        let status_bar = Paragraph::new(status_text)
            .style(Style::default().fg(Color::White))
            .block(Block::default().borders(Borders::ALL));
        f.render_widget(status_bar, chunks[2]);

        // Content based on selected tab
        match self.tab_index {
            0 => self.render_nodes_tab(f, chunks[1]),
            1 => self.render_contracts_tab(f, chunks[1]),
            2 => self.render_stats_tab(f, chunks[1]),
            _ => {}
        }
    }

    fn render_nodes_tab<B: Backend>(&self, f: &mut Frame<B>, area: Rect) {
        let network = self.network.lock().unwrap();

        // Create headers and rows for node table
        let header_cells = ["ID", "Layer", "Price", "Balance", "Comp Cost", "Queue"]
            .iter()
            .map(|h| Cell::from(*h).style(Style::default().fg(Color::Yellow)));
        let header = Row::new(header_cells).style(Style::default()).height(1);

        let rows = network.nodes.iter().enumerate().map(|(i, node)| {
            let is_selected = self.selected_node.map_or(false, |s| s == i);
            let style = if is_selected {
                Style::default().bg(Color::DarkGray)
            } else {
                Style::default()
            };

            let queue_time = network.time_in_queue(node.id);
            let cells = [
                Cell::from(node.id.to_string()),
                Cell::from(node.params.layer_range.to_string()),
                Cell::from(format!("{:.2}", node.price)),
                Cell::from(format!("{:.2}", node.balance)),
                Cell::from(node.params.computational_cost.to_string()),
                Cell::from(queue_time.to_string()),
            ];
            Row::new(cells).style(style)
        });

        let table = Table::new(rows)
            .header(header)
            .block(
                Block::default()
                    .title("Network Nodes")
                    .borders(Borders::ALL),
            )
            .highlight_style(Style::default().add_modifier(Modifier::BOLD))
            .widths(&[
                Constraint::Percentage(10),
                Constraint::Percentage(10),
                Constraint::Percentage(20),
                Constraint::Percentage(20),
                Constraint::Percentage(20),
                Constraint::Percentage(20),
            ]);

        f.render_widget(table, area);
    }

    fn render_contracts_tab<B: Backend>(&self, f: &mut Frame<B>, area: Rect) {
        let network = self.network.lock().unwrap();

        let active_contracts: Vec<ListItem> = network
            .contracts
            .iter()
            .enumerate()
            .filter(|(_, contract)| !contract.is_fulfilled())
            .map(|(i, contract)| {
                let text = vec![
                    Line::from(vec![
                        Span::raw(format!("Contract #{}: ", i)),
                        Span::styled(
                            format!("Owner: {}", Contract::get_owner(contract)),
                            Style::default().fg(Color::Yellow),
                        ),
                    ]),
                    Line::from(format!(
                        "   Subcontracts: {}",
                        Contract::num_subcontracts(contract)
                    )),
                ];
                ListItem::new(text)
            })
            .collect();

        let contracts_list = List::new(active_contracts)
            .block(
                Block::default()
                    .title("Active Contracts")
                    .borders(Borders::ALL),
            )
            .highlight_style(Style::default().add_modifier(Modifier::BOLD));

        f.render_widget(contracts_list, area);
    }

    fn render_stats_tab<B: Backend>(&self, f: &mut Frame<B>, area: Rect) {
        let network = self.network.lock().unwrap();

        // Calculate some statistics
        let total_nodes = network.nodes.len();
        let active_contracts = network
            .contracts
            .iter()
            .filter(|c| !c.is_fulfilled())
            .count();
        let completed_contracts = network
            .contracts
            .iter()
            .filter(|c| c.is_fulfilled())
            .count();

        // Calculate average node price and balance by layer
        let mut layer_stats: Vec<(u8, (f64, f64, usize))> = Vec::new();

        for layer in network.min_layer..=network.max_layer {
            let nodes_in_layer: Vec<&P2PNode> = network
                .nodes
                .iter()
                .filter(|n| n.params.layer_range == layer)
                .collect();

            let count = nodes_in_layer.len();
            if count > 0 {
                let avg_price = nodes_in_layer.iter().map(|n| n.price).sum::<f64>() / count as f64;
                let avg_balance =
                    nodes_in_layer.iter().map(|n| n.balance).sum::<f64>() / count as f64;
                layer_stats.push((layer, (avg_price, avg_balance, count)));
            }
        }

        // Create a paragraph with statistics
        let stats_text = vec![
            Line::from(vec![
                Span::raw("Total Nodes: "),
                Span::styled(total_nodes.to_string(), Style::default().fg(Color::Yellow)),
            ]),
            Line::from(vec![
                Span::raw("Active Contracts: "),
                Span::styled(
                    active_contracts.to_string(),
                    Style::default().fg(Color::Green),
                ),
            ]),
            Line::from(vec![
                Span::raw("Completed Contracts: "),
                Span::styled(
                    completed_contracts.to_string(),
                    Style::default().fg(Color::Cyan),
                ),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                "Layer Statistics:",
                Style::default().add_modifier(Modifier::BOLD),
            )),
        ];

        let mut all_stats = stats_text;
        for (layer, (avg_price, avg_balance, count)) in layer_stats {
            all_stats.push(Line::from(vec![
                Span::raw(format!("Layer {}: ", layer)),
                Span::styled(
                    format!("{} nodes", count),
                    Style::default().fg(Color::Yellow),
                ),
                Span::raw(" | Avg Price: "),
                Span::styled(
                    format!("{:.2}", avg_price),
                    Style::default().fg(Color::Green),
                ),
                Span::raw(" | Avg Balance: "),
                Span::styled(
                    format!("{:.2}", avg_balance),
                    Style::default().fg(Color::Cyan),
                ),
            ]));
        }

        let stats_widget = Paragraph::new(all_stats).block(
            Block::default()
                .title("Network Statistics")
                .borders(Borders::ALL),
        );

        f.render_widget(stats_widget, area);
    }
}

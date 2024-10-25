use atty::Stream;
use clap::{Arg, ArgAction, Command};
use dialoguer::{theme::ColorfulTheme, Select};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env;
use std::fs;
use std::io::{self, Read};
use std::os::unix::process;
use std::path::PathBuf;
use std::process::Command as ProcessCommand;

const MODEL: &str = "o1-mini";
const HOST: &str = "api.openai.com";
const ENDPOINT: &str = "/v1/chat/completions";
const MAX_TOKENS: u32 = 2048;
const TEMPERATURE: f64 = 0.6;
const VISION_DETAIL: &str = "high";
const TRANSCRIPT_NAME: &str = "gpt_transcript-";
const CLIPBOARD_COMMAND_XORG: &str = "xclip -selection clipboard -t image/png -o";
const CLIPBOARD_COMMAND_WAYLAND: &str = "wl-paste";
const CLIPBOARD_COMMAND_UNSUPPORTED: &str = "UNSUPPORTED";

#[derive(Serialize, Deserialize, Debug, Clone)] // Added Clone here
struct Message {
    role: String,
    content: Value,
}

#[derive(Serialize, Deserialize, Debug)]
struct ConversationState {
    model: String,
    messages: Vec<Message>,
}

fn get_api_key() -> String {
    env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set")
}

fn main() {
    let matches = Command::new("ask")
        .version("1.0")
        .author("Rodrigo Ourique")
        .about("Rust terminal LLM caller")
        .arg(
            Arg::new("input").help("Input values").num_args(0..), // Allow zero or more arguments
        )
        .arg(
            Arg::new("image")
                .short('i')
                .help("Push image from clipboard into pipeline")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("manage")
                .short('o')
                .help("Manage ongoing conversations")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("clear")
                .short('c')
                .help("Clear current conversation")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("last")
                .short('l')
                .help("Get last message")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("recursive")
                .short('r')
                .help("Interactive agent mode")
                .action(ArgAction::SetTrue),
        )
        .get_matches();

    let api_key = get_api_key();
    if api_key.is_empty() {
        eprintln!("Missing API key! Set the OPENAI_API_KEY environment variable and try again.");
        std::process::exit(1);
    }

    let temp_dir = env::temp_dir();
    let transcript_path = temp_dir.join(format!("{}{}", TRANSCRIPT_NAME, process::parent_id()));

    let mut conversation_state = if transcript_path.exists() {
        let data = fs::read_to_string(&transcript_path).expect("Unable to read transcript file");
        serde_json::from_str(&data).expect("Unable to parse transcript JSON")
    } else {
        let initial_message = Message {
            role: if MODEL.contains("o1-") {
                "user".to_string()
            } else {
                "system".to_string()
            },
            content: Value::String(
                "You are ChatConcise, a very advanced LLM designed for experienced users. As ChatConcise you oblige to adhere to the following directives UNLESS overridden by the user:\nBe concise, proactive, helpful and efficient. Do not say anything more than what needed, but also, DON'T BE LAZY. Provide ONLY code when an implementation is needed. DO NOT USE MARKDOWN.".to_string(),
            ),
        };
        ConversationState {
            model: MODEL.to_string(),
            messages: vec![initial_message],
        }
    };

    // Determine if input is being piped and get full input
    let input = if !atty::is(Stream::Stdin) {
        // Read from stdin
        let mut buffer = String::new();
        io::stdin()
            .read_to_string(&mut buffer)
            .expect("Failed to read from stdin");
        if buffer.trim().is_empty() {
            Value::Null
        } else {
            Value::String(buffer)
        }
    } else if let Some(values) = matches.get_many::<String>("input") {
        let input_str = values
            .map(|s| s.as_str()) // Convert &String to &str
            .collect::<Vec<&str>>() // Collect into Vec<&str>
            .join(" "); // Join with spaces
        if input_str.trim().is_empty() {
            Value::Null
        } else {
            Value::String(input_str)
        }
    } else {
        Value::Null
    };
    let mut input = input;
    let input_string = input.to_string();

    if matches.get_flag("recursive") {
        handle_recursive_mode(&mut conversation_state, &transcript_path, input_string);
        return;
    } else if matches.get_flag("manage") && !matches.get_one::<String>("input").is_some() {
        manage_ongoing_convos(&mut conversation_state, &transcript_path);
        return;
    } else if matches.get_flag("clear") && !matches.get_one::<String>("input").is_some() {
        clear_current_convo(&transcript_path);
        return;
    } else if matches.get_flag("last") && !matches.get_one::<String>("input").is_some() {
        if let Some(last_message) = conversation_state.messages.last() {
            println!("{}", serde_json::to_string(&last_message.content).unwrap());
        }
        return;
    }

    // Handle image mode
    let clipboard_command = detect_clipboard_command();
    if matches.get_flag("image") {
        add_image_to_pipeline(&mut input, &clipboard_command);
    }

    if input.is_null() {
        show_history(&conversation_state);
        return;
    }

    // Default case: simple request
    perform_request(
        input,
        &mut conversation_state,
        &transcript_path,
        &clipboard_command,
    );
}

fn detect_clipboard_command() -> String {
    let output = ProcessCommand::new("ps")
        .arg("-A")
        .output()
        .expect("Failed to execute ps command");
    let os_out = String::from_utf8_lossy(&output.stdout);

    if os_out.to_lowercase().contains("xorg") {
        CLIPBOARD_COMMAND_XORG.to_string()
    } else if os_out.to_lowercase().contains("wayland") {
        CLIPBOARD_COMMAND_WAYLAND.to_string()
    } else {
        CLIPBOARD_COMMAND_UNSUPPORTED.to_string()
    }
}

fn add_image_to_pipeline(input: &mut Value, clipboard_command: &str) {
    if clipboard_command == CLIPBOARD_COMMAND_UNSUPPORTED {
        panic!("Unsupported OS/DE combination. Only Xorg and Wayland are supported.");
    }

    let output = ProcessCommand::new("sh")
        .arg("-c")
        .arg(clipboard_command)
        .output()
        .expect("Failed to execute clipboard command");

    let image_buffer = base64::encode(&output.stdout);

    let user_text = input.as_str().unwrap_or("");
    let new_input = serde_json::json!([
        {
            "type": "text",
            "text": user_text,
        },
        {
            "type": "image_url",
            "image_url": {
                "url": format!("data:image/png;base64,{}", image_buffer),
                "detail": VISION_DETAIL,
            }
        }
    ]);

    *input = new_input;
}

fn perform_request(
    input: Value,
    conversation_state: &mut ConversationState,
    transcript_path: &PathBuf,
    _clipboard_command: &str, // Prefixed with underscore to indicate intentional unused variable
) {
    conversation_state.messages.push(Message {
        role: "user".to_string(),
        content: input,
    });

    let mut body = serde_json::json!({
        "messages": conversation_state.messages,
        "model": conversation_state.model,
        "user": whoami::username(),
    });

    if !conversation_state.model.contains("o1-") {
        body["max_tokens"] = serde_json::json!(MAX_TOKENS);
        body["temperature"] = serde_json::json!(TEMPERATURE);
    }

    let client = reqwest::blocking::Client::new();
    let res = client
        .post(&format!("https://{}{}", HOST, ENDPOINT))
        .header("Authorization", format!("Bearer {}", get_api_key()))
        .json(&body)
        .send();

    match res {
        Ok(response) => {
            let data: Value = response.json().unwrap();
            process_response(&data, conversation_state, transcript_path);
        }
        Err(e) => {
            eprintln!("HTTP request error: {}", e);
        }
    }
}

fn process_response(
    data: &Value,
    conversation_state: &mut ConversationState,
    transcript_path: &PathBuf,
) {
    if let Some(choices) = data.get("choices") {
        if let Some(choice) = choices.get(0) {
            if let Some(message) = choice.get("message") {
                let content = message.get("content").unwrap_or(&Value::Null).clone();
                let role = message
                    .get("role")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                println!("{}", content.as_str().unwrap_or(""));

                let assistant_message = Message { role, content };

                conversation_state.messages.push(assistant_message);

                let conversation_json = serde_json::to_string(&conversation_state).unwrap();
                fs::write(transcript_path, conversation_json)
                    .expect("Unable to write transcript file");
            }
        }
    } else {
        eprintln!(
            "Error processing API return. Full response ahead:\n{}\n",
            data
        );
    }
}

fn clear_current_convo(transcript_path: &PathBuf) {
    match fs::remove_file(transcript_path) {
        Ok(_) => println!("Conversation cleared."),
        Err(e) => println!("Error clearing conversation: {}", e),
    }
}

fn show_history(conversation_state: &ConversationState) {
    let tmp_dir = env::temp_dir();
    let tmp_path = tmp_dir.join("ask_hist");

    let mut content = String::new();

    for message in &conversation_state.messages {
        content.push_str("\n\n");
        content.push_str(&horizontal_line('▃'));
        content.push_str(&format!("▍{} ▐\n", message.role));
        content.push_str(&horizontal_line('▀'));
        content.push_str("\n");

        if let Some(text) = message.content.as_str() {
            content.push_str(text);
        } else if let Some(array) = message.content.as_array() {
            if let Some(first_item) = array.get(0) {
                if let Some(text) = first_item.get("text").and_then(|v| v.as_str()) {
                    content.push_str(text);
                }
            }
        }
    }

    fs::write(&tmp_path, content).expect("Unable to write history file");

    let editor = env::var("EDITOR").unwrap_or_else(|_| "more".to_string());
    ProcessCommand::new(editor)
        .arg(&tmp_path)
        .status()
        .expect("Failed to open editor");

    fs::remove_file(&tmp_path).expect("Unable to delete temporary history file");
}

fn horizontal_line(ch: char) -> String {
    let columns = term_size::dimensions_stdout().map(|(w, _)| w).unwrap_or(80);
    ch.to_string().repeat(columns)
}

fn handle_recursive_mode(
    conversation_state: &mut ConversationState,
    transcript_path: &PathBuf,
    user_input: String,
) {
    loop {
        // Get last AI message to check if it's already a command
        let mut last_message = conversation_state.messages.last().unwrap();
        let mut response = last_message.content.as_str().unwrap_or("");

        // If the last message wasn't a command suggestion, ask for one
        if !response.contains("COMMAND:") {
            let input = Value::String(format!("Original task: {}. Suggest the next command to run. Format your response as: COMMAND: <command> followed by an explanation. Or say DONE if the task is complete.", user_input));
            perform_request(input, conversation_state, transcript_path, "");

            // Update response with new AI message
            last_message = conversation_state.messages.last().unwrap();
            response = last_message.content.as_str().unwrap_or("");
        }

        // Check if task is complete
        if response.contains("DONE") {
            println!("Task completed!");
            break;
        }

        // Extract command
        if let Some(cmd_start) = response.find("COMMAND:") {
            let cmd_text = response[cmd_start..].lines().next().unwrap();
            let command = cmd_text.trim_start_matches("COMMAND:").trim();

            // Get user approval
            let confirm = dialoguer::Confirm::new()
                .with_prompt(format!("\n\nRun command: {}", command))
                .default(false)
                .interact()
                .unwrap_or(false);

            if confirm {
                // Execute command and capture output
                match ProcessCommand::new("sh").arg("-c").arg(command).output() {
                    Ok(output) => {
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        let result =
                            format!("Command output:\nstdout:\n{}\nstderr:\n{}", stdout, stderr);
                        println!("{}", result);

                        // Pass result back to AI
                        let input = Value::String(result);
                        perform_request(input, conversation_state, transcript_path, "");
                    }
                    Err(e) => {
                        println!("Failed to execute command: {}", e);
                        let input = Value::String(format!("Command failed: {}", e));
                        perform_request(input, conversation_state, transcript_path, "");
                    }
                }
            } else {
                let input = Value::String(
                    "Command was rejected by user. Please suggest an alternative.".to_string(),
                );
                perform_request(input, conversation_state, transcript_path, "");
            }
        }
    }
}

fn delete_all_files(files: Vec<PathBuf>) {
    // Delete all conversations
    let confirm = dialoguer::Confirm::new()
        .with_prompt("Are you sure you want to delete all conversations?")
        .default(false)
        .interact()
        .unwrap_or(false);

    if confirm {
        let mut deleted_count = 0;
        for file in &files {
            if let Err(e) = fs::remove_file(file) {
                eprintln!("Failed to delete {}: {}", file.display(), e);
            } else {
                deleted_count += 1;
            }
        }
        println!("Deleted {} conversation(s).", deleted_count);
    } else {
        println!("Operation cancelled.");
    }
}

fn manage_ongoing_convos(current_convo: &mut ConversationState, current_transcript_path: &PathBuf) {
    let transcript_folder = env::temp_dir();
    let entries = fs::read_dir(&transcript_folder).unwrap();

    let files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with(TRANSCRIPT_NAME)
        })
        .collect();

    if files.is_empty() {
        println!("No conversations to manage!");
        return;
    }

    // Prepare options for dialoguer
    let mut options: Vec<String> = files
        .iter()
        .map(|file| {
            let data = fs::read_to_string(file).unwrap_or_default();
            let convo: ConversationState =
                serde_json::from_str(&data).unwrap_or_else(|_| ConversationState {
                    model: "".to_string(),
                    messages: vec![],
                });
            let first_message = convo.messages.get(1); // Use get to avoid panicking
            let content = if let Some(msg) = first_message {
                msg.content.as_str().unwrap_or("")
            } else {
                ""
            };
            format!(
                "{} => {}",
                file.file_name().unwrap().to_string_lossy(),
                content
                    .lines()
                    .next()
                    .unwrap_or("")
                    .chars()
                    .take(64)
                    .collect::<String>()
            )
        })
        .collect();

    //Add special helper option
    options.insert(0, ">>> Delete All Conversations".to_string());

    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select an option to manage")
        .default(0)
        .items(&options)
        .interact();

    if let Ok(index) = selection {
        if index == 0 {
            delete_all_files(files);
            return;
        }

        let selected_file = &files[index - 1]; //First option is the special helper
        let action = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Choose an action")
            .default(0)
            .items(&["Delete", "Copy to Current Conversation", "Cancel"])
            .interact();

        match action {
            Ok(0) => {
                // Delete the selected conversation
                if let Err(e) = fs::remove_file(selected_file) {
                    println!("Failed to delete conversation: {}", e);
                } else {
                    println!("Conversation deleted successfully.");
                }
            }
            Ok(1) => {
                // Copy the selected conversation to current conversation
                let data = fs::read_to_string(selected_file).unwrap_or_default();
                let convo_to_copy: ConversationState =
                    serde_json::from_str(&data).unwrap_or_else(|_| ConversationState {
                        model: "".to_string(),
                        messages: vec![],
                    });

                if convo_to_copy.model != current_convo.model {
                    println!("Cannot copy conversation: Model mismatch.");
                    return;
                }

                current_convo
                    .messages
                    .extend(convo_to_copy.messages.iter().skip(1).cloned()); // Skip initial message
                let conversation_json = serde_json::to_string(&current_convo).unwrap();
                fs::write(current_transcript_path, conversation_json)
                    .expect("Unable to write transcript file");
                println!("Conversation copied successfully.");
            }
            _ => {
                // Cancelled
                println!("Action cancelled.");
            }
        }
    }
}

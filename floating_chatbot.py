import streamlit as st
import streamlit.components.v1 as components

# Initialize chat history for the components.html chatbot
if "floating_chat_history" not in st.session_state:
    st.session_state.floating_chat_history = ["🤖 Bot: Hello! How can I help you today?"]
if "floating_chat_input" not in st.session_state: # To capture input from JS
    st.session_state.floating_chat_input = ""

def floating_chatbot_ui():
    # Pass chat history and a way to send messages back to Python
    # This JSON.stringify is crucial to pass Python list to JS array
    # We need to ensure the string is properly escaped for HTML embedding
    chat_history_json = str(st.session_state.floating_chat_history)
    # Replace single quotes with double quotes for valid JSON, and escape existing double quotes
    chat_history_json = chat_history_json.replace("'", '"')
    chat_history_json = chat_history_json.replace('"', '\\"') # Escape internal double quotes for JS string literal

    components.html(f"""
    <style>
    /* Chat Button */
    #chat-button {{
        position: fixed !important;
        bottom: 30px; /* Adjusted for bottom-right placement */
        right: 30px; /* Adjusted for bottom-right placement */
        background-color: #25d366;
        color: white;
        border-radius: 50%;
        width: 55px;
        height: 55px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        cursor: pointer;
        z-index: 999999;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: background 0.3s;
    }}

    #chat-button.active {{
        background-color: #ff4d4d; /* red when chat is open */
    }}

    /* Chat Window */
    #chat-window {{
        position: fixed !important;
        bottom: 95px; /* Adjusted to sit above the button */
        right: 30px;
        width: 350px;
        height: 450px;
        background: #fff;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        display: none;
        flex-direction: column;
        z-index: 999998;
        overflow: hidden;
        font-family: sans-serif;
        color: #333;
    }}

    #chat-window.active {{ display: flex !important; }}

    #chat-header {{
        padding: 10px;
        font-weight: bold;
        border-bottom: 1px solid #ddd;
        background: #f1f1f1;
        display: flex;
        justify-content: center;
        align-items: center;
        color: #333;
    }}

    #chat-content {{
        flex: 1;
        padding: 10px;
        overflow-y: auto;
        font-size: 14px;
        color: #333;
    }}
    #chat-content p {{
        margin-bottom: 5px;
        line-height: 1.4;
    }}
    #chat-content p:last-child {{
        margin-bottom: 0;
    }}

    #chat-input-container {{
        border-top: 1px solid #ddd;
        padding: 10px;
        display: flex;
    }}

    #chat-input {{
        flex: 1;
        padding: 5px;
        font-size: 14px;
        border: 1px solid #ccc;
        border-radius: 6px;
        color: #333;
    }}

    #send-btn {{
        margin-left: 5px;
        background: #25d366;
        border: none;
        color: white;
        padding: 6px 12px;
        border-radius: 6px;
        cursor: pointer;
    }}

    #send-btn:hover {{
        background: #20b358;
    }}
    </style>

    <div id="chat-button">💬</div>

    <div id="chat-window">
        <div id="chat-header">AI Chatbot</div>
        <div id="chat-content">
        </div>
        <div id="chat-input-container">
            <input type="text" id="chat-input" placeholder="Type your message...">
            <button id="send-btn">Send</button>
        </div>
    </div>

    <script>
    const chatBtn = document.getElementById("chat-button");
    const chatWindow = document.getElementById("chat-window");
    const sendBtn = document.getElementById("send-btn");
    const input = document.getElementById("chat-input");
    const content = document.getElementById("chat-content");

    // Initialize chat history from Streamlit Python state
    let chatHistory = JSON.parse("{chat_history_json}");


    function renderChatHistory() {{
        content.innerHTML = ""; // Clear existing content
        chatHistory.forEach(msg => {{
            const p = document.createElement("p");
            p.textContent = msg;
            content.appendChild(p);
        }});
        content.scrollTop = content.scrollHeight;
    }}

    // Initial render when iframe loads
    renderChatHistory();

    chatBtn.onclick = () => {{
        chatWindow.classList.toggle("active");
        chatBtn.classList.toggle("active");
        chatBtn.textContent = chatWindow.classList.contains("active") ? "✖" : "💬";
        if (chatWindow.classList.contains("active")) {{
             content.scrollTop = content.scrollHeight;
        }}
    }}

    function sendMessage() {{
        const userMessageText = input.value.trim();
        if(userMessageText !== ""){{
            // Add user message to local JS history immediately
            const userMessage = "You: " + userMessageText;
            chatHistory.push(userMessage);
            renderChatHistory(); // Display immediately

            // IMPORTANT: Send message back to Streamlit parent via postMessage
            // This tells Streamlit to update its session state with the user's input
            window.parent.postMessage({{
                type: "streamlit:setComponentValue",
                componentId: "floating_chatbot_iframe", // Matches the key in components.html
                value: userMessageText
            }}, "*");

            input.value = ""; // Clear input field
        }}
    }}

    sendBtn.onclick = sendMessage;

    input.addEventListener("keypress", function(e){{
        if(e.key === "Enter"){{ e.preventDefault(); sendMessage(); }}
    }});
    </script>
    """, height=600, width=400, scrolling=False, key="floating_chatbot_iframe") # Add a key

    # After the components.html render, check if there's new input from JS
    # This part runs on every Streamlit rerun
    if st.session_state.floating_chat_input:
        user_query = st.session_state.floating_chat_input
        st.session_state.floating_chat_input = "" # Clear it immediately after reading

        # If the last message in history is not the one we just received (avoid duplicates on rerun)
        # This check is crucial because st.rerun makes things re-execute
        if not st.session_state.floating_chat_history or st.session_state.floating_chat_history[-1] != f"You: {user_query}":
             st.session_state.floating_chat_history.append(f"You: {user_query}")
        
        # We need to call the chatbot_response_generator here, but it's in main.py
        # This is why we need to make floating_chatbot_ui more integrated OR
        # pass the response_generator as a parameter.
        # For now, let's assume `main.py` will handle generating the response
        # and updating `st.session_state.floating_chat_history` before rerunning.
        # This means `floating_chatbot_ui` itself won't generate the bot's reply.
        # It just receives it when it rerenders with the updated `floating_chat_history`.

        # Instead of generating a response here, we just triggered a rerun.
        # The main app will detect the input and generate the response.
        # This function's job is just to embed the UI and send the message.
        st.experimental_set_query_params(floating_chat_message=user_query) # A way to signal main app
        st.rerun() # Rerun the main Streamlit app to process the message
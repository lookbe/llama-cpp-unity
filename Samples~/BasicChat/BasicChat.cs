using LlamaCpp;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class BasicChat : MonoBehaviour
{
    public LlamaCpp.ChatCompletion chatbot;

    public TMP_Text chatHistory;
    public TMP_InputField chatInputField;
    public Button sendButton;

    void Start()
    {
        chatbot.InitModel();
        sendButton.onClick.AddListener(OnSendButtonClicked);
    }

    private void OnEnable()
    {
        if (chatbot != null)
        {
            chatbot.OnResponseStreamed += OnBotResponseStreamed;
            chatbot.OnResponseGenerated += OnBotResponseGenerated;
            chatbot.OnStatusChanged += OnBotStatusChanged;

            OnBotStatusChanged(chatbot.status);
        }
    }

    private void OnDisable()
    {
        if (chatbot != null)
        {
            chatbot.OnStatusChanged -= OnBotStatusChanged;
            chatbot.OnResponseGenerated -= OnBotResponseGenerated;
            chatbot.OnResponseStreamed -= OnBotResponseStreamed;
        }
    }

    void OnBotStatusChanged(ChatCompletion.Status status)
    {
        switch (status)
        {
            case ChatCompletion.Status.Loading:
                {
                    sendButton.interactable = false;
                }
                break;
            case ChatCompletion.Status.Ready:
                {
                    sendButton.interactable = true;

                    ClearInput();
                }
                break;
            case ChatCompletion.Status.Generate:
                {
                    sendButton.interactable = false;
                }
                break;
            case ChatCompletion.Status.Error:
                {
                    sendButton.interactable = true;
                }
                break;
        }
    }

    void OnBotResponseStreamed(string response)
    {
        chatHistory.text += response;
    }

    void OnBotResponseGenerated(string response)
    {
        Debug.Log(response);
        chatHistory.text += '\n';
    }

    protected virtual void ClearInput()
    {
        chatInputField.text = "";
    }

    public void OnSendButtonClicked()
    {
        if (chatbot)
        {
            string message = chatInputField.text;
            if (!string.IsNullOrEmpty(message))
            {
                chatHistory.text += "user: " + message + "\nbot: ";
                chatbot.Prompt(message);
                ClearInput();
            }
        }
    }
}

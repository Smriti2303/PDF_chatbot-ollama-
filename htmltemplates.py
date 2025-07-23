css = """
<style>
body {
    background-image: url("https://static.vecteezy.com/system/resources/previews/029/693/717/large_2x/robot-chatbot-head-icon-sign-made-with-binary-code-in-wireframe-hand-chatbot-assistant-application-digital-binary-data-and-streaming-digital-code-background-with-digits-1-0-illustration-vector.jpg"); 
    background-size: cover;
    background-attachment: fixed;
    color: #E5E5E5;
}

main {
    background-color: rgba(10, 10, 10, 0.85);
    padding: 1rem;
    border-radius: 10px;
}

h1, h2, h3 {
    color: #FFBF00;
}

section[data-testid="stSidebar"] {
    background-color: #111;
    color: white;
}

input[type="text"], textarea {
    background-color: #222 !important;
    color: white !important;
    border: 1px solid #444 !important;
}
</style>
"""

bot_template = """
<div style='background-color:#222222;padding:10px;border-radius:10px;margin-bottom:10px;border-left:5px solid #FFA500;'>
    <img src='https://img.freepik.com/premium-photo/female-robot-with-humanoid-face_1252540-77.jpg' width='32' style='vertical-align:middle;margin-right:10px'>
    <span style='color:white;'>{{MSG}}</span>
</div>
"""

user_template = """
<div style='background-color:#333333;padding:10px;border-radius:10px;margin-bottom:10px;border-left:5px solid #1E90FF;'>
    <img src='https://th.bing.com/th/id/OIP.MAreZc3ERyeUmCIz3t5EaQHaHa?w=700&h=700&rs=1&pid=ImgDetMain' width='32' style='vertical-align:middle;margin-right:10px'>
    <span style='color:white;'>{{MSG}}</span>
</div>
"""

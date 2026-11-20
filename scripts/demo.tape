# Renders the demo video with VHS (https://github.com/charmbracelet/vhs): no screen capture.
#   docker compose down -v && docker compose up --build -d   # Nimbus must start on v1
#   vhs scripts/demo.tape                                     # -> recordings/demo.mp4
# Uses ~5 Groq calls. Timings leave time to read each answer; total ~2 minutes.

Output recordings/demo.mp4
Output recordings/demo.gif

Set Shell "bash"
Set FontSize 22
Set Width 1920
Set Height 1080
Set Padding 40
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms

# Setup off camera: mint tokens, check Nimbus is on v1, warm up.
Hide
Type "clear && DEMO_CLEAR=1 scripts/demo.sh"
Enter
Wait+Screen@60s /Ready/
Sleep 500ms
Show

Sleep 2s
Enter
# 1. health
Sleep 5s
Enter
# 2. escort 7:45 PM on v1
Wait+Screen@15s /latency_ms/
Sleep 12s
Enter
# 3. pet dog, 0 LLM calls
Wait+Screen@15s /latency_ms/
Sleep 9s
Enter
# 4. Kestrel asks about Nimbus
Wait+Screen@15s /latency_ms/
Sleep 9s
Enter
# 5. Saffron planted instruction
Wait+Screen@15s /latency_ms/
Sleep 11s
Enter
# 6. company_id in body -> 422
Wait+Screen@15s /detail/
Sleep 7s
Enter
# 7. employee on admin -> 403
Wait+Screen@15s /detail/
Sleep 6s
Enter
# 8. admin uploads v2 -> 201
Wait+Screen@30s /chunk_count/
Sleep 9s
Enter
# 9. active v2, v1 retired
Wait+Screen@15s /active_version/
Sleep 8s
Enter
# 10. escort 7:45 PM on v2
Wait+Screen@15s /latency_ms/
Sleep 12s
Enter
# 11. eval headline
Wait+Screen@15s /EVAL_REPORT/
Sleep 12s

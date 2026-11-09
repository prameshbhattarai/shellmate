"""`shellmate init zsh`: a key binding that sends what you typed at the prompt to shellmate.

Add to ~/.zshrc:   eval "$(shellmate init zsh)"
Then type a request at the prompt and press Ctrl-G instead of Enter.
"""

ZSH = r"""# shellmate: press Ctrl-G to send the current command line to shellmate as a request
_shellmate_widget() {
  local request="$BUFFER"
  [[ -z "${request// }" ]] && return
  BUFFER=""
  zle -I                       # let shellmate draw below the prompt
  print -s -- "# shellmate: $request"
  shellmate ask "$request" </dev/tty
  zle reset-prompt
}
zle -N _shellmate_widget
bindkey '^G' _shellmate_widget
"""

SCRIPTS = {"zsh": ZSH}

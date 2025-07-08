# Claude Code Notes

Do not mock responses. API calls should fail loudly. If you catch an error and return mock data, I don't know when things are broken! I have asked you to stop doing this, please, listen.

Testing scripts are encouraged. Put them in scripts/, where you can leave them as reminders for yourself.

Always lint your code, `./lint.sh [check|fix|format]` — use a SubAgent / Task for this so you don't waste your tokens on linting errors.

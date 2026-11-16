package main

import (
	"flag"
	"os"
	"time"
)

const rowSchema = "pw-decide-row/2"
const retentionRowSchema = "pw-retention-row/1"

func runDecideCorpus(args []string) error {
	fs := flag.NewFlagSet("decide corpus", flag.ContinueOnError)
	out := fs.String("out", "", "output file (default stdout)")
	return write(*out)
}

func runDecideExport(args []string) error {
	fs := flag.NewFlagSet("decide export", flag.ContinueOnError)
	dbPath := fs.String("db", "", "path to the sidecar store.db (required)")
	outPath := fs.String("out", "", "output file (default stdout)")
	after := fs.Int64("after", 0, "export receipts with an id greater than this")
	retentionPath := fs.String("retention", "", "write compaction observation rows to this file")
	requestsPath := fs.String("requests", "", "write conversation usage rows to this file")
	rootsPath := fs.String("roots", "", "file of root session ids")
	row := map[string]any{"schema": "pw-request-row/1"}
	return export(*dbPath, *outPath, *after, *retentionPath, *requestsPath, *rootsPath, row)
}

func runDecideGenerate(args []string) error {
	fs := flag.NewFlagSet("decide generate", flag.ContinueOnError)
	addr := fs.String("addr", "", "sidecar URL")
	token := fs.String("token", os.Getenv("LYCAON_API_TOKEN"), "sidecar API token")
	tasksPath := fs.String("tasks", "", "JSON-lines task file (required)")
	project := fs.String("project", "", "project folder the sessions work in (required)")
	projectName := fs.String("project-name", "", "project name recorded in the store")
	manifestPath := fs.String("manifest", "", "manifest output, one JSON line per task (required)")
	policyPath := fs.String("unattended", "", "unattended policy YAML (required)")
	timeout := fs.Duration("timeout", 15*time.Minute, "per-prompt timeout")
	return generate(*addr, *token, *tasksPath, *project, *projectName, *manifestPath, *policyPath, *timeout)
}

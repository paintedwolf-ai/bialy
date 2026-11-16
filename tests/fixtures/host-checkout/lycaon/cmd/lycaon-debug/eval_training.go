package main

import "flag"

func runEvalTrainingContract(args []string) error {
	flags := flag.NewFlagSet("eval training-contract", flag.ContinueOnError)
	out := flags.String("out", "", "new directory for contract.json")
	return contract(*out)
}

func runEvalTrainingBundle(args []string) error {
	flags := flag.NewFlagSet("eval training-bundle", flag.ContinueOnError)
	var req BundleRequest
	flags.StringVar(&req.Capture, "capture", "", "closed application capture directory")
	flags.StringVar(&req.Captures, "captures", "", "training capture directory of that application")
	flags.StringVar(&req.Project, "project", "", "retained delivered project directory")
	flags.StringVar(&req.SessionID, "session", "", "root session id")
	flags.StringVar(&req.Out, "out", "", "new directory for the bundle")
	return bundle(req)
}

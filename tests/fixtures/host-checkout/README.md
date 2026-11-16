A reduced Painted Wolf Code checkout: only the declarations `bialy.host_contract`
reads from a host checkout, in the places the host keeps them. The contract tests
verify the recorded contract against this tree and against copies with one
declaration removed.

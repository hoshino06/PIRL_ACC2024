AgentDir = "TD20Lam1e-4_S5000_B09";
%AgentDir = "RS_C005";

tic
parfor AgentIdx = 1:8

    main_acc2024(AgentIdx, AgentDir)
    %main_reward_shaping(AgentIdx, AgentDir)

end
toc

// Fill out your copyright notice in the Description page of Project Settings.

using UnrealBuildTool;
using System.Collections.Generic;

public class CH7_CharacterIKTarget : TargetRules
{
	public CH7_CharacterIKTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Game;

		ExtraModuleNames.AddRange( new string[] { "CH7_CharacterIK" } );
	}
}

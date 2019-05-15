// Fill out your copyright notice in the Description page of Project Settings.

using UnrealBuildTool;
using System.Collections.Generic;

public class CH7_CharacterIKEditorTarget : TargetRules
{
	public CH7_CharacterIKEditorTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Editor;

		ExtraModuleNames.AddRange( new string[] { "CH7_CharacterIK" } );
	}
}

#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.7125595155788095,-0.1552669782807973,-1.4863179224105554>, 1 }        
    sphere {  m*<0.3429181349127381,0.2866693419916973,8.447984035456322>, 1 }
    sphere {  m*<4.273158454271334,0.02480205662208998,-3.881214027074787>, 1 }
    sphere {  m*<-2.3561348389025922,2.1734842605762967,-2.422037717012771>, 1}
    sphere { m*<-2.088347617864761,-2.7142076818276006,-2.2324914318502005>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3429181349127381,0.2866693419916973,8.447984035456322>, <-0.7125595155788095,-0.1552669782807973,-1.4863179224105554>, 0.5 }
    cylinder { m*<4.273158454271334,0.02480205662208998,-3.881214027074787>, <-0.7125595155788095,-0.1552669782807973,-1.4863179224105554>, 0.5}
    cylinder { m*<-2.3561348389025922,2.1734842605762967,-2.422037717012771>, <-0.7125595155788095,-0.1552669782807973,-1.4863179224105554>, 0.5 }
    cylinder {  m*<-2.088347617864761,-2.7142076818276006,-2.2324914318502005>, <-0.7125595155788095,-0.1552669782807973,-1.4863179224105554>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.7125595155788095,-0.1552669782807973,-1.4863179224105554>, 1 }        
    sphere {  m*<0.3429181349127381,0.2866693419916973,8.447984035456322>, 1 }
    sphere {  m*<4.273158454271334,0.02480205662208998,-3.881214027074787>, 1 }
    sphere {  m*<-2.3561348389025922,2.1734842605762967,-2.422037717012771>, 1}
    sphere { m*<-2.088347617864761,-2.7142076818276006,-2.2324914318502005>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3429181349127381,0.2866693419916973,8.447984035456322>, <-0.7125595155788095,-0.1552669782807973,-1.4863179224105554>, 0.5 }
    cylinder { m*<4.273158454271334,0.02480205662208998,-3.881214027074787>, <-0.7125595155788095,-0.1552669782807973,-1.4863179224105554>, 0.5}
    cylinder { m*<-2.3561348389025922,2.1734842605762967,-2.422037717012771>, <-0.7125595155788095,-0.1552669782807973,-1.4863179224105554>, 0.5 }
    cylinder {  m*<-2.088347617864761,-2.7142076818276006,-2.2324914318502005>, <-0.7125595155788095,-0.1552669782807973,-1.4863179224105554>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
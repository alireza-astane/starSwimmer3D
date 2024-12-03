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
    sphere { m*<-0.915181806637462,-0.16260648328826247,-1.389541410528141>, 1 }        
    sphere {  m*<0.24631294569282264,0.2846610876691225,8.532677765954013>, 1 }
    sphere {  m*<4.992201897721973,0.04773781739789251,-4.3050541153015045>, 1 }
    sphere {  m*<-2.567684485701503,2.166283382232,-2.3090516027264814>, 1}
    sphere { m*<-2.2998972646636715,-2.7214085601718976,-2.119505317563911>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.24631294569282264,0.2846610876691225,8.532677765954013>, <-0.915181806637462,-0.16260648328826247,-1.389541410528141>, 0.5 }
    cylinder { m*<4.992201897721973,0.04773781739789251,-4.3050541153015045>, <-0.915181806637462,-0.16260648328826247,-1.389541410528141>, 0.5}
    cylinder { m*<-2.567684485701503,2.166283382232,-2.3090516027264814>, <-0.915181806637462,-0.16260648328826247,-1.389541410528141>, 0.5 }
    cylinder {  m*<-2.2998972646636715,-2.7214085601718976,-2.119505317563911>, <-0.915181806637462,-0.16260648328826247,-1.389541410528141>, 0.5}

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
    sphere { m*<-0.915181806637462,-0.16260648328826247,-1.389541410528141>, 1 }        
    sphere {  m*<0.24631294569282264,0.2846610876691225,8.532677765954013>, 1 }
    sphere {  m*<4.992201897721973,0.04773781739789251,-4.3050541153015045>, 1 }
    sphere {  m*<-2.567684485701503,2.166283382232,-2.3090516027264814>, 1}
    sphere { m*<-2.2998972646636715,-2.7214085601718976,-2.119505317563911>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.24631294569282264,0.2846610876691225,8.532677765954013>, <-0.915181806637462,-0.16260648328826247,-1.389541410528141>, 0.5 }
    cylinder { m*<4.992201897721973,0.04773781739789251,-4.3050541153015045>, <-0.915181806637462,-0.16260648328826247,-1.389541410528141>, 0.5}
    cylinder { m*<-2.567684485701503,2.166283382232,-2.3090516027264814>, <-0.915181806637462,-0.16260648328826247,-1.389541410528141>, 0.5 }
    cylinder {  m*<-2.2998972646636715,-2.7214085601718976,-2.119505317563911>, <-0.915181806637462,-0.16260648328826247,-1.389541410528141>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
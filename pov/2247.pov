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
    sphere { m*<1.0875293592909554,0.3292808298675945,0.5088870347519463>, 1 }        
    sphere {  m*<1.3316128879114504,0.35497973078781253,3.498829547922221>, 1 }
    sphere {  m*<3.8248600769739847,0.3549797307878125,-0.718452660568397>, 1 }
    sphere {  m*<-3.1052960981844726,7.005214206208854,-1.9701842589688645>, 1}
    sphere { m*<-3.7769951975203218,-7.920443566022668,-2.366658724434548>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3316128879114504,0.35497973078781253,3.498829547922221>, <1.0875293592909554,0.3292808298675945,0.5088870347519463>, 0.5 }
    cylinder { m*<3.8248600769739847,0.3549797307878125,-0.718452660568397>, <1.0875293592909554,0.3292808298675945,0.5088870347519463>, 0.5}
    cylinder { m*<-3.1052960981844726,7.005214206208854,-1.9701842589688645>, <1.0875293592909554,0.3292808298675945,0.5088870347519463>, 0.5 }
    cylinder {  m*<-3.7769951975203218,-7.920443566022668,-2.366658724434548>, <1.0875293592909554,0.3292808298675945,0.5088870347519463>, 0.5}

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
    sphere { m*<1.0875293592909554,0.3292808298675945,0.5088870347519463>, 1 }        
    sphere {  m*<1.3316128879114504,0.35497973078781253,3.498829547922221>, 1 }
    sphere {  m*<3.8248600769739847,0.3549797307878125,-0.718452660568397>, 1 }
    sphere {  m*<-3.1052960981844726,7.005214206208854,-1.9701842589688645>, 1}
    sphere { m*<-3.7769951975203218,-7.920443566022668,-2.366658724434548>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3316128879114504,0.35497973078781253,3.498829547922221>, <1.0875293592909554,0.3292808298675945,0.5088870347519463>, 0.5 }
    cylinder { m*<3.8248600769739847,0.3549797307878125,-0.718452660568397>, <1.0875293592909554,0.3292808298675945,0.5088870347519463>, 0.5}
    cylinder { m*<-3.1052960981844726,7.005214206208854,-1.9701842589688645>, <1.0875293592909554,0.3292808298675945,0.5088870347519463>, 0.5 }
    cylinder {  m*<-3.7769951975203218,-7.920443566022668,-2.366658724434548>, <1.0875293592909554,0.3292808298675945,0.5088870347519463>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
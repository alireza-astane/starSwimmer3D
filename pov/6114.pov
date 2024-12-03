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
    sphere { m*<-1.5213483510240717,-0.3072327645752915,-0.9896692107779602>, 1 }        
    sphere {  m*<-0.05899991287013484,0.1865670399533163,8.890585352556176>, 1 }
    sphere {  m*<7.2963515251298405,0.09764676395895916,-5.6889079374891836>, 1 }
    sphere {  m*<-3.6856148440738234,2.6189365688480155,-2.099470330005466>, 1}
    sphere { m*<-2.8989285187935057,-2.883890320050905,-1.66894249712841>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.05899991287013484,0.1865670399533163,8.890585352556176>, <-1.5213483510240717,-0.3072327645752915,-0.9896692107779602>, 0.5 }
    cylinder { m*<7.2963515251298405,0.09764676395895916,-5.6889079374891836>, <-1.5213483510240717,-0.3072327645752915,-0.9896692107779602>, 0.5}
    cylinder { m*<-3.6856148440738234,2.6189365688480155,-2.099470330005466>, <-1.5213483510240717,-0.3072327645752915,-0.9896692107779602>, 0.5 }
    cylinder {  m*<-2.8989285187935057,-2.883890320050905,-1.66894249712841>, <-1.5213483510240717,-0.3072327645752915,-0.9896692107779602>, 0.5}

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
    sphere { m*<-1.5213483510240717,-0.3072327645752915,-0.9896692107779602>, 1 }        
    sphere {  m*<-0.05899991287013484,0.1865670399533163,8.890585352556176>, 1 }
    sphere {  m*<7.2963515251298405,0.09764676395895916,-5.6889079374891836>, 1 }
    sphere {  m*<-3.6856148440738234,2.6189365688480155,-2.099470330005466>, 1}
    sphere { m*<-2.8989285187935057,-2.883890320050905,-1.66894249712841>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.05899991287013484,0.1865670399533163,8.890585352556176>, <-1.5213483510240717,-0.3072327645752915,-0.9896692107779602>, 0.5 }
    cylinder { m*<7.2963515251298405,0.09764676395895916,-5.6889079374891836>, <-1.5213483510240717,-0.3072327645752915,-0.9896692107779602>, 0.5}
    cylinder { m*<-3.6856148440738234,2.6189365688480155,-2.099470330005466>, <-1.5213483510240717,-0.3072327645752915,-0.9896692107779602>, 0.5 }
    cylinder {  m*<-2.8989285187935057,-2.883890320050905,-1.66894249712841>, <-1.5213483510240717,-0.3072327645752915,-0.9896692107779602>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
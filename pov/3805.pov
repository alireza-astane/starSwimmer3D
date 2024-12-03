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
    sphere { m*<-0.04622167137097488,0.12050295959325186,-0.15483100185980259>, 1 }        
    sphere {  m*<0.19451343337071664,0.24921303777357728,2.8327237692607476>, 1 }
    sphere {  m*<2.6884867226352878,0.22253693497962646,-1.3840405273109893>, 1 }
    sphere {  m*<-1.6678370312638666,2.4489769040118547,-1.1287767672757747>, 1}
    sphere { m*<-2.012882681081079,-3.5971877188133776,-1.2943019801745619>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.19451343337071664,0.24921303777357728,2.8327237692607476>, <-0.04622167137097488,0.12050295959325186,-0.15483100185980259>, 0.5 }
    cylinder { m*<2.6884867226352878,0.22253693497962646,-1.3840405273109893>, <-0.04622167137097488,0.12050295959325186,-0.15483100185980259>, 0.5}
    cylinder { m*<-1.6678370312638666,2.4489769040118547,-1.1287767672757747>, <-0.04622167137097488,0.12050295959325186,-0.15483100185980259>, 0.5 }
    cylinder {  m*<-2.012882681081079,-3.5971877188133776,-1.2943019801745619>, <-0.04622167137097488,0.12050295959325186,-0.15483100185980259>, 0.5}

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
    sphere { m*<-0.04622167137097488,0.12050295959325186,-0.15483100185980259>, 1 }        
    sphere {  m*<0.19451343337071664,0.24921303777357728,2.8327237692607476>, 1 }
    sphere {  m*<2.6884867226352878,0.22253693497962646,-1.3840405273109893>, 1 }
    sphere {  m*<-1.6678370312638666,2.4489769040118547,-1.1287767672757747>, 1}
    sphere { m*<-2.012882681081079,-3.5971877188133776,-1.2943019801745619>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.19451343337071664,0.24921303777357728,2.8327237692607476>, <-0.04622167137097488,0.12050295959325186,-0.15483100185980259>, 0.5 }
    cylinder { m*<2.6884867226352878,0.22253693497962646,-1.3840405273109893>, <-0.04622167137097488,0.12050295959325186,-0.15483100185980259>, 0.5}
    cylinder { m*<-1.6678370312638666,2.4489769040118547,-1.1287767672757747>, <-0.04622167137097488,0.12050295959325186,-0.15483100185980259>, 0.5 }
    cylinder {  m*<-2.012882681081079,-3.5971877188133776,-1.2943019801745619>, <-0.04622167137097488,0.12050295959325186,-0.15483100185980259>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
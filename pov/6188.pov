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
    sphere { m*<-1.4724525842455292,-0.38769640261020866,-0.9645085253595682>, 1 }        
    sphere {  m*<-0.013016414576548785,0.1474015799464129,8.914024364195548>, 1 }
    sphere {  m*<7.342335023423424,0.058481303952055835,-5.665468925849808>, 1 }
    sphere {  m*<-3.952608045811429,2.9169521019526257,-2.2359953563275274>, 1}
    sphere { m*<-2.8335112934259072,-2.9753130742785006,-1.6353794064577625>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.013016414576548785,0.1474015799464129,8.914024364195548>, <-1.4724525842455292,-0.38769640261020866,-0.9645085253595682>, 0.5 }
    cylinder { m*<7.342335023423424,0.058481303952055835,-5.665468925849808>, <-1.4724525842455292,-0.38769640261020866,-0.9645085253595682>, 0.5}
    cylinder { m*<-3.952608045811429,2.9169521019526257,-2.2359953563275274>, <-1.4724525842455292,-0.38769640261020866,-0.9645085253595682>, 0.5 }
    cylinder {  m*<-2.8335112934259072,-2.9753130742785006,-1.6353794064577625>, <-1.4724525842455292,-0.38769640261020866,-0.9645085253595682>, 0.5}

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
    sphere { m*<-1.4724525842455292,-0.38769640261020866,-0.9645085253595682>, 1 }        
    sphere {  m*<-0.013016414576548785,0.1474015799464129,8.914024364195548>, 1 }
    sphere {  m*<7.342335023423424,0.058481303952055835,-5.665468925849808>, 1 }
    sphere {  m*<-3.952608045811429,2.9169521019526257,-2.2359953563275274>, 1}
    sphere { m*<-2.8335112934259072,-2.9753130742785006,-1.6353794064577625>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.013016414576548785,0.1474015799464129,8.914024364195548>, <-1.4724525842455292,-0.38769640261020866,-0.9645085253595682>, 0.5 }
    cylinder { m*<7.342335023423424,0.058481303952055835,-5.665468925849808>, <-1.4724525842455292,-0.38769640261020866,-0.9645085253595682>, 0.5}
    cylinder { m*<-3.952608045811429,2.9169521019526257,-2.2359953563275274>, <-1.4724525842455292,-0.38769640261020866,-0.9645085253595682>, 0.5 }
    cylinder {  m*<-2.8335112934259072,-2.9753130742785006,-1.6353794064577625>, <-1.4724525842455292,-0.38769640261020866,-0.9645085253595682>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
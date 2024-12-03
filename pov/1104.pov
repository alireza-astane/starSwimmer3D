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
    sphere { m*<0.14702877010050608,-4.166240917902427e-18,1.1436780156840305>, 1 }        
    sphere {  m*<0.16631102193248826,-3.4912331626249595e-18,4.143616637949863>, 1 }
    sphere {  m*<8.862111250288686,2.626574726284225e-18,-2.0007772410701072>, 1 }
    sphere {  m*<-4.588344716263031,8.164965809277259,-2.1592879845514714>, 1}
    sphere { m*<-4.588344716263031,-8.164965809277259,-2.159287984551474>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.16631102193248826,-3.4912331626249595e-18,4.143616637949863>, <0.14702877010050608,-4.166240917902427e-18,1.1436780156840305>, 0.5 }
    cylinder { m*<8.862111250288686,2.626574726284225e-18,-2.0007772410701072>, <0.14702877010050608,-4.166240917902427e-18,1.1436780156840305>, 0.5}
    cylinder { m*<-4.588344716263031,8.164965809277259,-2.1592879845514714>, <0.14702877010050608,-4.166240917902427e-18,1.1436780156840305>, 0.5 }
    cylinder {  m*<-4.588344716263031,-8.164965809277259,-2.159287984551474>, <0.14702877010050608,-4.166240917902427e-18,1.1436780156840305>, 0.5}

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
    sphere { m*<0.14702877010050608,-4.166240917902427e-18,1.1436780156840305>, 1 }        
    sphere {  m*<0.16631102193248826,-3.4912331626249595e-18,4.143616637949863>, 1 }
    sphere {  m*<8.862111250288686,2.626574726284225e-18,-2.0007772410701072>, 1 }
    sphere {  m*<-4.588344716263031,8.164965809277259,-2.1592879845514714>, 1}
    sphere { m*<-4.588344716263031,-8.164965809277259,-2.159287984551474>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.16631102193248826,-3.4912331626249595e-18,4.143616637949863>, <0.14702877010050608,-4.166240917902427e-18,1.1436780156840305>, 0.5 }
    cylinder { m*<8.862111250288686,2.626574726284225e-18,-2.0007772410701072>, <0.14702877010050608,-4.166240917902427e-18,1.1436780156840305>, 0.5}
    cylinder { m*<-4.588344716263031,8.164965809277259,-2.1592879845514714>, <0.14702877010050608,-4.166240917902427e-18,1.1436780156840305>, 0.5 }
    cylinder {  m*<-4.588344716263031,-8.164965809277259,-2.159287984551474>, <0.14702877010050608,-4.166240917902427e-18,1.1436780156840305>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
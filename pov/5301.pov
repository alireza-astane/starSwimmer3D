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
    sphere { m*<-0.6338895188422198,-0.15234760556255456,-1.5220890322007825>, 1 }        
    sphere {  m*<0.3786885975874947,0.28741883589562534,8.41677632965932>, 1 }
    sphere {  m*<3.9828930150130466,0.01534152900869784,-3.715333829585128>, 1 }
    sphere {  m*<-2.2738063647355387,2.1763513132408234,-2.4643355141531824>, 1}
    sphere { m*<-2.006019143697707,-2.711340629163074,-2.274789228990612>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3786885975874947,0.28741883589562534,8.41677632965932>, <-0.6338895188422198,-0.15234760556255456,-1.5220890322007825>, 0.5 }
    cylinder { m*<3.9828930150130466,0.01534152900869784,-3.715333829585128>, <-0.6338895188422198,-0.15234760556255456,-1.5220890322007825>, 0.5}
    cylinder { m*<-2.2738063647355387,2.1763513132408234,-2.4643355141531824>, <-0.6338895188422198,-0.15234760556255456,-1.5220890322007825>, 0.5 }
    cylinder {  m*<-2.006019143697707,-2.711340629163074,-2.274789228990612>, <-0.6338895188422198,-0.15234760556255456,-1.5220890322007825>, 0.5}

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
    sphere { m*<-0.6338895188422198,-0.15234760556255456,-1.5220890322007825>, 1 }        
    sphere {  m*<0.3786885975874947,0.28741883589562534,8.41677632965932>, 1 }
    sphere {  m*<3.9828930150130466,0.01534152900869784,-3.715333829585128>, 1 }
    sphere {  m*<-2.2738063647355387,2.1763513132408234,-2.4643355141531824>, 1}
    sphere { m*<-2.006019143697707,-2.711340629163074,-2.274789228990612>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3786885975874947,0.28741883589562534,8.41677632965932>, <-0.6338895188422198,-0.15234760556255456,-1.5220890322007825>, 0.5 }
    cylinder { m*<3.9828930150130466,0.01534152900869784,-3.715333829585128>, <-0.6338895188422198,-0.15234760556255456,-1.5220890322007825>, 0.5}
    cylinder { m*<-2.2738063647355387,2.1763513132408234,-2.4643355141531824>, <-0.6338895188422198,-0.15234760556255456,-1.5220890322007825>, 0.5 }
    cylinder {  m*<-2.006019143697707,-2.711340629163074,-2.274789228990612>, <-0.6338895188422198,-0.15234760556255456,-1.5220890322007825>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
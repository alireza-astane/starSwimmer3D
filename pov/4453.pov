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
    sphere { m*<-0.1973454969333874,-0.10072943342798071,-0.8047848328194576>, 1 }        
    sphere {  m*<0.29840832876466916,0.16432752090470223,5.347586292508128>, 1 }
    sphere {  m*<2.5373628970728697,0.001304541958393482,-2.033994358270639>, 1 }
    sphere {  m*<-1.8189608568262774,2.227744510990618,-1.7787305982354258>, 1}
    sphere { m*<-1.5511736357884456,-2.659947431413279,-1.589184313072853>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.29840832876466916,0.16432752090470223,5.347586292508128>, <-0.1973454969333874,-0.10072943342798071,-0.8047848328194576>, 0.5 }
    cylinder { m*<2.5373628970728697,0.001304541958393482,-2.033994358270639>, <-0.1973454969333874,-0.10072943342798071,-0.8047848328194576>, 0.5}
    cylinder { m*<-1.8189608568262774,2.227744510990618,-1.7787305982354258>, <-0.1973454969333874,-0.10072943342798071,-0.8047848328194576>, 0.5 }
    cylinder {  m*<-1.5511736357884456,-2.659947431413279,-1.589184313072853>, <-0.1973454969333874,-0.10072943342798071,-0.8047848328194576>, 0.5}

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
    sphere { m*<-0.1973454969333874,-0.10072943342798071,-0.8047848328194576>, 1 }        
    sphere {  m*<0.29840832876466916,0.16432752090470223,5.347586292508128>, 1 }
    sphere {  m*<2.5373628970728697,0.001304541958393482,-2.033994358270639>, 1 }
    sphere {  m*<-1.8189608568262774,2.227744510990618,-1.7787305982354258>, 1}
    sphere { m*<-1.5511736357884456,-2.659947431413279,-1.589184313072853>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.29840832876466916,0.16432752090470223,5.347586292508128>, <-0.1973454969333874,-0.10072943342798071,-0.8047848328194576>, 0.5 }
    cylinder { m*<2.5373628970728697,0.001304541958393482,-2.033994358270639>, <-0.1973454969333874,-0.10072943342798071,-0.8047848328194576>, 0.5}
    cylinder { m*<-1.8189608568262774,2.227744510990618,-1.7787305982354258>, <-0.1973454969333874,-0.10072943342798071,-0.8047848328194576>, 0.5 }
    cylinder {  m*<-1.5511736357884456,-2.659947431413279,-1.589184313072853>, <-0.1973454969333874,-0.10072943342798071,-0.8047848328194576>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
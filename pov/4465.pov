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
    sphere { m*<-0.19877819955817247,-0.10149543416162157,-0.8225648635584986>, 1 }        
    sphere {  m*<0.30371620907265157,0.16716540236011054,5.413457795360463>, 1 }
    sphere {  m*<2.5359301944480848,0.0005385412247526367,-2.05177438900968>, 1 }
    sphere {  m*<-1.8203935594510625,2.226978510256978,-1.7965106289744668>, 1}
    sphere { m*<-1.5526063384132307,-2.6607134321469195,-1.606964343811894>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.30371620907265157,0.16716540236011054,5.413457795360463>, <-0.19877819955817247,-0.10149543416162157,-0.8225648635584986>, 0.5 }
    cylinder { m*<2.5359301944480848,0.0005385412247526367,-2.05177438900968>, <-0.19877819955817247,-0.10149543416162157,-0.8225648635584986>, 0.5}
    cylinder { m*<-1.8203935594510625,2.226978510256978,-1.7965106289744668>, <-0.19877819955817247,-0.10149543416162157,-0.8225648635584986>, 0.5 }
    cylinder {  m*<-1.5526063384132307,-2.6607134321469195,-1.606964343811894>, <-0.19877819955817247,-0.10149543416162157,-0.8225648635584986>, 0.5}

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
    sphere { m*<-0.19877819955817247,-0.10149543416162157,-0.8225648635584986>, 1 }        
    sphere {  m*<0.30371620907265157,0.16716540236011054,5.413457795360463>, 1 }
    sphere {  m*<2.5359301944480848,0.0005385412247526367,-2.05177438900968>, 1 }
    sphere {  m*<-1.8203935594510625,2.226978510256978,-1.7965106289744668>, 1}
    sphere { m*<-1.5526063384132307,-2.6607134321469195,-1.606964343811894>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.30371620907265157,0.16716540236011054,5.413457795360463>, <-0.19877819955817247,-0.10149543416162157,-0.8225648635584986>, 0.5 }
    cylinder { m*<2.5359301944480848,0.0005385412247526367,-2.05177438900968>, <-0.19877819955817247,-0.10149543416162157,-0.8225648635584986>, 0.5}
    cylinder { m*<-1.8203935594510625,2.226978510256978,-1.7965106289744668>, <-0.19877819955817247,-0.10149543416162157,-0.8225648635584986>, 0.5 }
    cylinder {  m*<-1.5526063384132307,-2.6607134321469195,-1.606964343811894>, <-0.19877819955817247,-0.10149543416162157,-0.8225648635584986>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
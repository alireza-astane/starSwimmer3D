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
    sphere { m*<0.5643201357818811,1.1078699318596188,0.19953399801964944>, 1 }        
    sphere {  m*<0.8057138425743865,1.2149938939928806,3.187882844734241>, 1 }
    sphere {  m*<3.298961031636921,1.2149938939928802,-1.0293993637563739>, 1 }
    sphere {  m*<-1.3375783546901743,3.854500871309691,-0.9249802383218064>, 1}
    sphere { m*<-3.960844979704824,-7.3985523037912015,-2.475372391385946>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8057138425743865,1.2149938939928806,3.187882844734241>, <0.5643201357818811,1.1078699318596188,0.19953399801964944>, 0.5 }
    cylinder { m*<3.298961031636921,1.2149938939928802,-1.0293993637563739>, <0.5643201357818811,1.1078699318596188,0.19953399801964944>, 0.5}
    cylinder { m*<-1.3375783546901743,3.854500871309691,-0.9249802383218064>, <0.5643201357818811,1.1078699318596188,0.19953399801964944>, 0.5 }
    cylinder {  m*<-3.960844979704824,-7.3985523037912015,-2.475372391385946>, <0.5643201357818811,1.1078699318596188,0.19953399801964944>, 0.5}

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
    sphere { m*<0.5643201357818811,1.1078699318596188,0.19953399801964944>, 1 }        
    sphere {  m*<0.8057138425743865,1.2149938939928806,3.187882844734241>, 1 }
    sphere {  m*<3.298961031636921,1.2149938939928802,-1.0293993637563739>, 1 }
    sphere {  m*<-1.3375783546901743,3.854500871309691,-0.9249802383218064>, 1}
    sphere { m*<-3.960844979704824,-7.3985523037912015,-2.475372391385946>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8057138425743865,1.2149938939928806,3.187882844734241>, <0.5643201357818811,1.1078699318596188,0.19953399801964944>, 0.5 }
    cylinder { m*<3.298961031636921,1.2149938939928802,-1.0293993637563739>, <0.5643201357818811,1.1078699318596188,0.19953399801964944>, 0.5}
    cylinder { m*<-1.3375783546901743,3.854500871309691,-0.9249802383218064>, <0.5643201357818811,1.1078699318596188,0.19953399801964944>, 0.5 }
    cylinder {  m*<-3.960844979704824,-7.3985523037912015,-2.475372391385946>, <0.5643201357818811,1.1078699318596188,0.19953399801964944>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
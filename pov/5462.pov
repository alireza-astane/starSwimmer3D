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
    sphere { m*<-0.8474791338690613,-0.16018196821946634,-1.42259772067785>, 1 }        
    sphere {  m*<0.27933206604573774,0.28534636820975473,8.503700706227718>, 1 }
    sphere {  m*<4.755764089636717,0.040271299950693146,-4.163739351076254>, 1 }
    sphere {  m*<-2.4970790797179636,2.168661081705438,-2.347423489903151>, 1}
    sphere { m*<-2.2292918586801322,-2.7190308606984592,-2.1578772047405805>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.27933206604573774,0.28534636820975473,8.503700706227718>, <-0.8474791338690613,-0.16018196821946634,-1.42259772067785>, 0.5 }
    cylinder { m*<4.755764089636717,0.040271299950693146,-4.163739351076254>, <-0.8474791338690613,-0.16018196821946634,-1.42259772067785>, 0.5}
    cylinder { m*<-2.4970790797179636,2.168661081705438,-2.347423489903151>, <-0.8474791338690613,-0.16018196821946634,-1.42259772067785>, 0.5 }
    cylinder {  m*<-2.2292918586801322,-2.7190308606984592,-2.1578772047405805>, <-0.8474791338690613,-0.16018196821946634,-1.42259772067785>, 0.5}

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
    sphere { m*<-0.8474791338690613,-0.16018196821946634,-1.42259772067785>, 1 }        
    sphere {  m*<0.27933206604573774,0.28534636820975473,8.503700706227718>, 1 }
    sphere {  m*<4.755764089636717,0.040271299950693146,-4.163739351076254>, 1 }
    sphere {  m*<-2.4970790797179636,2.168661081705438,-2.347423489903151>, 1}
    sphere { m*<-2.2292918586801322,-2.7190308606984592,-2.1578772047405805>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.27933206604573774,0.28534636820975473,8.503700706227718>, <-0.8474791338690613,-0.16018196821946634,-1.42259772067785>, 0.5 }
    cylinder { m*<4.755764089636717,0.040271299950693146,-4.163739351076254>, <-0.8474791338690613,-0.16018196821946634,-1.42259772067785>, 0.5}
    cylinder { m*<-2.4970790797179636,2.168661081705438,-2.347423489903151>, <-0.8474791338690613,-0.16018196821946634,-1.42259772067785>, 0.5 }
    cylinder {  m*<-2.2292918586801322,-2.7190308606984592,-2.1578772047405805>, <-0.8474791338690613,-0.16018196821946634,-1.42259772067785>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
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
    sphere { m*<-0.19580078783861327,-0.09990354794456703,-0.7856147868428331>, 1 }        
    sphere {  m*<0.29265073968227867,0.16124920068331275,5.276133843760762>, 1 }
    sphere {  m*<2.5389076061676437,0.0021304274418071784,-2.014824312294014>, 1 }
    sphere {  m*<-1.8174161477315032,2.2285703964740318,-1.7595605522588011>, 1}
    sphere { m*<-1.5496289266936714,-2.6591215459298656,-1.5700142670962283>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.29265073968227867,0.16124920068331275,5.276133843760762>, <-0.19580078783861327,-0.09990354794456703,-0.7856147868428331>, 0.5 }
    cylinder { m*<2.5389076061676437,0.0021304274418071784,-2.014824312294014>, <-0.19580078783861327,-0.09990354794456703,-0.7856147868428331>, 0.5}
    cylinder { m*<-1.8174161477315032,2.2285703964740318,-1.7595605522588011>, <-0.19580078783861327,-0.09990354794456703,-0.7856147868428331>, 0.5 }
    cylinder {  m*<-1.5496289266936714,-2.6591215459298656,-1.5700142670962283>, <-0.19580078783861327,-0.09990354794456703,-0.7856147868428331>, 0.5}

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
    sphere { m*<-0.19580078783861327,-0.09990354794456703,-0.7856147868428331>, 1 }        
    sphere {  m*<0.29265073968227867,0.16124920068331275,5.276133843760762>, 1 }
    sphere {  m*<2.5389076061676437,0.0021304274418071784,-2.014824312294014>, 1 }
    sphere {  m*<-1.8174161477315032,2.2285703964740318,-1.7595605522588011>, 1}
    sphere { m*<-1.5496289266936714,-2.6591215459298656,-1.5700142670962283>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.29265073968227867,0.16124920068331275,5.276133843760762>, <-0.19580078783861327,-0.09990354794456703,-0.7856147868428331>, 0.5 }
    cylinder { m*<2.5389076061676437,0.0021304274418071784,-2.014824312294014>, <-0.19580078783861327,-0.09990354794456703,-0.7856147868428331>, 0.5}
    cylinder { m*<-1.8174161477315032,2.2285703964740318,-1.7595605522588011>, <-0.19580078783861327,-0.09990354794456703,-0.7856147868428331>, 0.5 }
    cylinder {  m*<-1.5496289266936714,-2.6591215459298656,-1.5700142670962283>, <-0.19580078783861327,-0.09990354794456703,-0.7856147868428331>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
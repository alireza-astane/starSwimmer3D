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
    sphere { m*<-1.1455641966776304,-0.8569218835105692,-0.7967188217144433>, 1 }        
    sphere {  m*<0.2946054824392075,-0.10883597014443838,9.070792146448172>, 1 }
    sphere {  m*<7.649956920439174,-0.19775624613879533,-5.508701143597172>, 1 }
    sphere {  m*<-5.518749269594635,4.549033353356048,-3.0361211810138338>, 1}
    sphere { m*<-2.4201375709564656,-3.499040204738095,-1.4236229163671146>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2946054824392075,-0.10883597014443838,9.070792146448172>, <-1.1455641966776304,-0.8569218835105692,-0.7967188217144433>, 0.5 }
    cylinder { m*<7.649956920439174,-0.19775624613879533,-5.508701143597172>, <-1.1455641966776304,-0.8569218835105692,-0.7967188217144433>, 0.5}
    cylinder { m*<-5.518749269594635,4.549033353356048,-3.0361211810138338>, <-1.1455641966776304,-0.8569218835105692,-0.7967188217144433>, 0.5 }
    cylinder {  m*<-2.4201375709564656,-3.499040204738095,-1.4236229163671146>, <-1.1455641966776304,-0.8569218835105692,-0.7967188217144433>, 0.5}

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
    sphere { m*<-1.1455641966776304,-0.8569218835105692,-0.7967188217144433>, 1 }        
    sphere {  m*<0.2946054824392075,-0.10883597014443838,9.070792146448172>, 1 }
    sphere {  m*<7.649956920439174,-0.19775624613879533,-5.508701143597172>, 1 }
    sphere {  m*<-5.518749269594635,4.549033353356048,-3.0361211810138338>, 1}
    sphere { m*<-2.4201375709564656,-3.499040204738095,-1.4236229163671146>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2946054824392075,-0.10883597014443838,9.070792146448172>, <-1.1455641966776304,-0.8569218835105692,-0.7967188217144433>, 0.5 }
    cylinder { m*<7.649956920439174,-0.19775624613879533,-5.508701143597172>, <-1.1455641966776304,-0.8569218835105692,-0.7967188217144433>, 0.5}
    cylinder { m*<-5.518749269594635,4.549033353356048,-3.0361211810138338>, <-1.1455641966776304,-0.8569218835105692,-0.7967188217144433>, 0.5 }
    cylinder {  m*<-2.4201375709564656,-3.499040204738095,-1.4236229163671146>, <-1.1455641966776304,-0.8569218835105692,-0.7967188217144433>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
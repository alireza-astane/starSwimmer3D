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
    sphere { m*<-1.1185735041147742,-0.8919339224249313,-0.7828874743694823>, 1 }        
    sphere {  m*<0.3200878143749706,-0.12992534219808513,9.08377741238513>, 1 }
    sphere {  m*<7.675439252374939,-0.21884561819244186,-5.495715877660206>, 1 }
    sphere {  m*<-5.63795174352944,4.667481009573708,-3.0969852454251443>, 1}
    sphere { m*<-2.3874185127474528,-3.5374966371315306,-1.4068804356389693>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3200878143749706,-0.12992534219808513,9.08377741238513>, <-1.1185735041147742,-0.8919339224249313,-0.7828874743694823>, 0.5 }
    cylinder { m*<7.675439252374939,-0.21884561819244186,-5.495715877660206>, <-1.1185735041147742,-0.8919339224249313,-0.7828874743694823>, 0.5}
    cylinder { m*<-5.63795174352944,4.667481009573708,-3.0969852454251443>, <-1.1185735041147742,-0.8919339224249313,-0.7828874743694823>, 0.5 }
    cylinder {  m*<-2.3874185127474528,-3.5374966371315306,-1.4068804356389693>, <-1.1185735041147742,-0.8919339224249313,-0.7828874743694823>, 0.5}

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
    sphere { m*<-1.1185735041147742,-0.8919339224249313,-0.7828874743694823>, 1 }        
    sphere {  m*<0.3200878143749706,-0.12992534219808513,9.08377741238513>, 1 }
    sphere {  m*<7.675439252374939,-0.21884561819244186,-5.495715877660206>, 1 }
    sphere {  m*<-5.63795174352944,4.667481009573708,-3.0969852454251443>, 1}
    sphere { m*<-2.3874185127474528,-3.5374966371315306,-1.4068804356389693>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3200878143749706,-0.12992534219808513,9.08377741238513>, <-1.1185735041147742,-0.8919339224249313,-0.7828874743694823>, 0.5 }
    cylinder { m*<7.675439252374939,-0.21884561819244186,-5.495715877660206>, <-1.1185735041147742,-0.8919339224249313,-0.7828874743694823>, 0.5}
    cylinder { m*<-5.63795174352944,4.667481009573708,-3.0969852454251443>, <-1.1185735041147742,-0.8919339224249313,-0.7828874743694823>, 0.5 }
    cylinder {  m*<-2.3874185127474528,-3.5374966371315306,-1.4068804356389693>, <-1.1185735041147742,-0.8919339224249313,-0.7828874743694823>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
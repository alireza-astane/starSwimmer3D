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
    sphere { m*<-0.03919243093767033,0.13379073084927895,-0.1507583043111142>, 1 }        
    sphere {  m*<0.2015426738040214,0.26250080902960427,2.8367964668094365>, 1 }
    sphere {  m*<2.695515963068592,0.23582470623565355,-1.3799678297623008>, 1 }
    sphere {  m*<-1.6608077908305616,2.462264675267882,-1.1247040697270863>, 1}
    sphere { m*<-2.0437606285357166,-3.6555580515009387,-1.3121924680729693>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2015426738040214,0.26250080902960427,2.8367964668094365>, <-0.03919243093767033,0.13379073084927895,-0.1507583043111142>, 0.5 }
    cylinder { m*<2.695515963068592,0.23582470623565355,-1.3799678297623008>, <-0.03919243093767033,0.13379073084927895,-0.1507583043111142>, 0.5}
    cylinder { m*<-1.6608077908305616,2.462264675267882,-1.1247040697270863>, <-0.03919243093767033,0.13379073084927895,-0.1507583043111142>, 0.5 }
    cylinder {  m*<-2.0437606285357166,-3.6555580515009387,-1.3121924680729693>, <-0.03919243093767033,0.13379073084927895,-0.1507583043111142>, 0.5}

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
    sphere { m*<-0.03919243093767033,0.13379073084927895,-0.1507583043111142>, 1 }        
    sphere {  m*<0.2015426738040214,0.26250080902960427,2.8367964668094365>, 1 }
    sphere {  m*<2.695515963068592,0.23582470623565355,-1.3799678297623008>, 1 }
    sphere {  m*<-1.6608077908305616,2.462264675267882,-1.1247040697270863>, 1}
    sphere { m*<-2.0437606285357166,-3.6555580515009387,-1.3121924680729693>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2015426738040214,0.26250080902960427,2.8367964668094365>, <-0.03919243093767033,0.13379073084927895,-0.1507583043111142>, 0.5 }
    cylinder { m*<2.695515963068592,0.23582470623565355,-1.3799678297623008>, <-0.03919243093767033,0.13379073084927895,-0.1507583043111142>, 0.5}
    cylinder { m*<-1.6608077908305616,2.462264675267882,-1.1247040697270863>, <-0.03919243093767033,0.13379073084927895,-0.1507583043111142>, 0.5 }
    cylinder {  m*<-2.0437606285357166,-3.6555580515009387,-1.3121924680729693>, <-0.03919243093767033,0.13379073084927895,-0.1507583043111142>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
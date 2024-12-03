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
    sphere { m*<-0.19509048695735337,-0.09952378246725281,-0.7767998581309774>, 1 }        
    sphere {  m*<0.289990749097155,0.15982702506610244,5.243123005676932>, 1 }
    sphere {  m*<2.5396179070489038,0.0025101929191214123,-2.006009383582158>, 1 }
    sphere {  m*<-1.8167058468502433,2.228950161951346,-1.7507456235469452>, 1}
    sphere { m*<-1.5489186258124115,-2.6587417804525515,-1.5611993383843723>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.289990749097155,0.15982702506610244,5.243123005676932>, <-0.19509048695735337,-0.09952378246725281,-0.7767998581309774>, 0.5 }
    cylinder { m*<2.5396179070489038,0.0025101929191214123,-2.006009383582158>, <-0.19509048695735337,-0.09952378246725281,-0.7767998581309774>, 0.5}
    cylinder { m*<-1.8167058468502433,2.228950161951346,-1.7507456235469452>, <-0.19509048695735337,-0.09952378246725281,-0.7767998581309774>, 0.5 }
    cylinder {  m*<-1.5489186258124115,-2.6587417804525515,-1.5611993383843723>, <-0.19509048695735337,-0.09952378246725281,-0.7767998581309774>, 0.5}

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
    sphere { m*<-0.19509048695735337,-0.09952378246725281,-0.7767998581309774>, 1 }        
    sphere {  m*<0.289990749097155,0.15982702506610244,5.243123005676932>, 1 }
    sphere {  m*<2.5396179070489038,0.0025101929191214123,-2.006009383582158>, 1 }
    sphere {  m*<-1.8167058468502433,2.228950161951346,-1.7507456235469452>, 1}
    sphere { m*<-1.5489186258124115,-2.6587417804525515,-1.5611993383843723>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.289990749097155,0.15982702506610244,5.243123005676932>, <-0.19509048695735337,-0.09952378246725281,-0.7767998581309774>, 0.5 }
    cylinder { m*<2.5396179070489038,0.0025101929191214123,-2.006009383582158>, <-0.19509048695735337,-0.09952378246725281,-0.7767998581309774>, 0.5}
    cylinder { m*<-1.8167058468502433,2.228950161951346,-1.7507456235469452>, <-0.19509048695735337,-0.09952378246725281,-0.7767998581309774>, 0.5 }
    cylinder {  m*<-1.5489186258124115,-2.6587417804525515,-1.5611993383843723>, <-0.19509048695735337,-0.09952378246725281,-0.7767998581309774>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
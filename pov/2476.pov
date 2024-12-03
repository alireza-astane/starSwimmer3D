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
    sphere { m*<0.908051152325684,0.6171855253522236,0.4027677473007658>, 1 }        
    sphere {  m*<1.151625692790351,0.6692280353880672,3.3924082316357813>, 1 }
    sphere {  m*<3.6448728818528875,0.669228035388067,-0.8248739768548359>, 1 }
    sphere {  m*<-2.5416135714279986,5.9377603252339135,-1.6368912618406097>, 1}
    sphere { m*<-3.8478839114970813,-7.717567286409695,-2.40857655958552>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.151625692790351,0.6692280353880672,3.3924082316357813>, <0.908051152325684,0.6171855253522236,0.4027677473007658>, 0.5 }
    cylinder { m*<3.6448728818528875,0.669228035388067,-0.8248739768548359>, <0.908051152325684,0.6171855253522236,0.4027677473007658>, 0.5}
    cylinder { m*<-2.5416135714279986,5.9377603252339135,-1.6368912618406097>, <0.908051152325684,0.6171855253522236,0.4027677473007658>, 0.5 }
    cylinder {  m*<-3.8478839114970813,-7.717567286409695,-2.40857655958552>, <0.908051152325684,0.6171855253522236,0.4027677473007658>, 0.5}

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
    sphere { m*<0.908051152325684,0.6171855253522236,0.4027677473007658>, 1 }        
    sphere {  m*<1.151625692790351,0.6692280353880672,3.3924082316357813>, 1 }
    sphere {  m*<3.6448728818528875,0.669228035388067,-0.8248739768548359>, 1 }
    sphere {  m*<-2.5416135714279986,5.9377603252339135,-1.6368912618406097>, 1}
    sphere { m*<-3.8478839114970813,-7.717567286409695,-2.40857655958552>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.151625692790351,0.6692280353880672,3.3924082316357813>, <0.908051152325684,0.6171855253522236,0.4027677473007658>, 0.5 }
    cylinder { m*<3.6448728818528875,0.669228035388067,-0.8248739768548359>, <0.908051152325684,0.6171855253522236,0.4027677473007658>, 0.5}
    cylinder { m*<-2.5416135714279986,5.9377603252339135,-1.6368912618406097>, <0.908051152325684,0.6171855253522236,0.4027677473007658>, 0.5 }
    cylinder {  m*<-3.8478839114970813,-7.717567286409695,-2.40857655958552>, <0.908051152325684,0.6171855253522236,0.4027677473007658>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    
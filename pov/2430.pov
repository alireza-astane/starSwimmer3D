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
    sphere { m*<0.9442059165705801,0.5608685185861904,0.42414474404707736>, 1 }        
    sphere {  m*<1.1879145766975863,0.6074391880116583,3.41386475146363>, 1 }
    sphere {  m*<3.6811617657601228,0.6074391880116581,-0.8034174570269892>, 1 }
    sphere {  m*<-2.6570115746503635,6.1519080015841725,-1.7051233602981697>, 1}
    sphere { m*<-3.834378964510478,-7.756378533035693,-2.40059082257964>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1879145766975863,0.6074391880116583,3.41386475146363>, <0.9442059165705801,0.5608685185861904,0.42414474404707736>, 0.5 }
    cylinder { m*<3.6811617657601228,0.6074391880116581,-0.8034174570269892>, <0.9442059165705801,0.5608685185861904,0.42414474404707736>, 0.5}
    cylinder { m*<-2.6570115746503635,6.1519080015841725,-1.7051233602981697>, <0.9442059165705801,0.5608685185861904,0.42414474404707736>, 0.5 }
    cylinder {  m*<-3.834378964510478,-7.756378533035693,-2.40059082257964>, <0.9442059165705801,0.5608685185861904,0.42414474404707736>, 0.5}

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
    sphere { m*<0.9442059165705801,0.5608685185861904,0.42414474404707736>, 1 }        
    sphere {  m*<1.1879145766975863,0.6074391880116583,3.41386475146363>, 1 }
    sphere {  m*<3.6811617657601228,0.6074391880116581,-0.8034174570269892>, 1 }
    sphere {  m*<-2.6570115746503635,6.1519080015841725,-1.7051233602981697>, 1}
    sphere { m*<-3.834378964510478,-7.756378533035693,-2.40059082257964>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1879145766975863,0.6074391880116583,3.41386475146363>, <0.9442059165705801,0.5608685185861904,0.42414474404707736>, 0.5 }
    cylinder { m*<3.6811617657601228,0.6074391880116581,-0.8034174570269892>, <0.9442059165705801,0.5608685185861904,0.42414474404707736>, 0.5}
    cylinder { m*<-2.6570115746503635,6.1519080015841725,-1.7051233602981697>, <0.9442059165705801,0.5608685185861904,0.42414474404707736>, 0.5 }
    cylinder {  m*<-3.834378964510478,-7.756378533035693,-2.40059082257964>, <0.9442059165705801,0.5608685185861904,0.42414474404707736>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    